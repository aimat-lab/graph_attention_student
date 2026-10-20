"""
ONNX-exportable stand-ins for the attention layers.

The layers in ``torch/layers.py`` are :class:`torch_geometric.nn.MessagePassing`
subclasses. ``propagate()`` is generated at runtime by PyG's code generator, and
the layers additionally stash their attention logits on ``self`` inside
``message()`` and read them back after ``propagate()`` returns. Neither survives
tracing.

The classes here reimplement the same arithmetic as an explicit
gather -> elementwise -> scatter sequence. They **wrap** a trained layer and reuse
its submodules by reference, so there is exactly one copy of the weights and no
state to keep in sync - only the plumbing is reimplemented. Output is bit-identical
to the PyG original in eval mode (see ``tests/test_torch_export.py``).

PyG's ``flow='source_to_target'`` convention, which these follow: ``x_j`` is the
source node (``edge_index[0]``), ``x_i`` is the target (``edge_index[1]``), and the
aggregation scatters into ``edge_index[1]``.
"""
import typing as t

import torch
import torch.nn as nn

from graph_attention_student.torch.layers import (
    GraphAttentionLayer,
    GraphAttentionLayerV2,
    GraphAttentionLayerV3,
    ParallelHeadAttention,
)
from graph_attention_student.torch.export.scatter_ops import segment_sum

__all__ = [
    'ExplicitGraphAttentionLayer',
    'ExplicitGraphAttentionLayerV2',
    'ExplicitGraphAttentionLayerV3',
    'ExplicitParallelHeadAttention',
    'UnsupportedLayerError',
    'explicit_layer_from',
]


class UnsupportedLayerError(TypeError):
    """Raised when a layer has no ONNX-exportable stand-in."""


class ExplicitGraphAttentionLayer(nn.Module):
    """Explicit form of :class:`GraphAttentionLayer` (``layer_version='v1'``)."""

    def __init__(self, layer: GraphAttentionLayer):
        super().__init__()
        self.layer = layer

    def forward(self, x, edge_attr, edge_index, edge_weights=None, **kwargs):
        lay = self.layer
        src, dst = edge_index[0], edge_index[1]

        message = torch.cat([x[dst], x[src], edge_attr], dim=-1)
        logits = lay.lay_attention(message)
        result = torch.sigmoid(logits) * lay.lay_linear(x[src])

        node_embedding = segment_sum(result, dst, x.shape[0])
        node_embedding = lay.lay_act(node_embedding)
        node_embedding = lay.lay_transform(torch.cat([node_embedding, x], dim=-1))
        return node_embedding, logits


class ExplicitGraphAttentionLayerV2(nn.Module):
    """
    Explicit form of :class:`GraphAttentionLayerV2` (``layer_version='v2'``, the default).

    Note that ``x_neighbor`` is the ``SumNeighbors`` sub-layer, itself a
    MessagePassing module, which is why there are two segment sums here.
    """

    def __init__(self, layer: GraphAttentionLayerV2):
        super().__init__()
        self.layer = layer

    def forward(self, x, edge_attr, edge_index, edge_weights=None, **kwargs):
        lay = self.layer
        src, dst = edge_index[0], edge_index[1]

        # SumNeighbors(x, edge_index)
        x_neighbor = segment_sum(x[src], dst, x.shape[0])

        message = torch.cat([x[dst], x[src], x_neighbor[src], edge_attr], dim=-1)
        logits = lay.lay_attention(message)
        result = torch.sigmoid(logits) * lay.lay_message(message)
        if edge_weights is not None:
            result = result * edge_weights

        node_embedding = segment_sum(result, dst, x.shape[0])
        node_embedding = lay.lay_transform(torch.cat([node_embedding, x], dim=-1))
        node_embedding = node_embedding + x
        return node_embedding, logits


class ExplicitGraphAttentionLayerV3(nn.Module):
    """
    Explicit form of :class:`GraphAttentionLayerV3` (``layer_version='v3'``).

    V3 ignores ``edge_weights`` - it swallows them in ``**kwargs`` - so the
    ``edge_importance_overwrite`` argument has no effect here, matching the original.
    """

    def __init__(self, layer: GraphAttentionLayerV3):
        super().__init__()
        self.layer = layer

    def forward(self, x, edge_attr, edge_index, edge_weights=None, **kwargs):
        lay = self.layer
        src, dst = edge_index[0], edge_index[1]

        message = torch.cat([x[dst], x[src], edge_attr], dim=-1)
        logits = lay.lay_attention(message)

        results = torch.stack([m(message) for m in lay.message_layers], dim=-1)
        result = torch.sigmoid(logits) * torch.mean(results, dim=-1)

        node_embedding = segment_sum(result, dst, x.shape[0])
        node_embedding = lay.lay_act(node_embedding)
        node_embedding = lay.lay_transform(torch.cat([node_embedding, x], dim=-1))
        return torch.tanh(node_embedding), logits


class ExplicitParallelHeadAttention(nn.Module):
    """
    Explicit form of :class:`ParallelHeadAttention`, the per-channel wrapper that
    ``Megan`` actually places into ``self.encoder_layers``.
    """

    def __init__(self, layer: ParallelHeadAttention):
        super().__init__()
        self.layers = nn.ModuleList([explicit_layer_from(lay) for lay in layer.layers])

    def forward(self, x, edge_attr, edge_index, edge_weights=None, **kwargs):
        node_embeddings: t.List[torch.Tensor] = []
        alphas: t.List[torch.Tensor] = []

        for channel_index, lay in enumerate(self.layers):
            weights = None
            if edge_weights is not None:
                weights = edge_weights[:, channel_index].unsqueeze(-1)

            node_embedding, alpha = lay(
                x[:, :, channel_index],
                edge_attr,
                edge_index,
                edge_weights=weights,
            )
            node_embeddings.append(node_embedding)
            alphas.append(alpha)

        return torch.stack(node_embeddings, dim=-1), torch.cat(alphas, dim=-1)


#: Maps each supported layer class onto its explicit stand-in.
EXPLICIT_LAYER_MAP: t.Dict[type, type] = {
    ParallelHeadAttention: ExplicitParallelHeadAttention,
    GraphAttentionLayer: ExplicitGraphAttentionLayer,
    GraphAttentionLayerV2: ExplicitGraphAttentionLayerV2,
    GraphAttentionLayerV3: ExplicitGraphAttentionLayerV3,
}


def explicit_layer_from(layer: nn.Module) -> nn.Module:
    """
    Wraps ``layer`` in its ONNX-exportable stand-in, sharing the original's weights.

    :raises UnsupportedLayerError: if there is no stand-in for the layer's type.
    """
    # An exact type lookup rather than isinstance, because V2 and V3 are siblings and a
    # subclass would silently get the wrong arithmetic.
    try:
        explicit_class = EXPLICIT_LAYER_MAP[type(layer)]
    except KeyError:
        raise UnsupportedLayerError(
            f'The layer "{type(layer).__name__}" has no ONNX-exportable equivalent. '
            f'Supported layers: {", ".join(c.__name__ for c in EXPLICIT_LAYER_MAP)}. '
            f'If this is a new attention layer, add a stand-in for it in '
            f'graph_attention_student/torch/export/layers_explicit.py.'
        )

    return explicit_class(layer)
