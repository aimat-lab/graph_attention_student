"""
The tensor-in / tuple-out facade that is actually traced by the ONNX exporter.

ONNX has no notion of a PyG ``Data`` object or of a dict return value, so this
module adapts ``Megan.forward`` at both ends without duplicating any of it:

- Inputs arrive as plain tensors and are bundled into a ``SimpleNamespace``.
  ``Megan.forward`` reads only ``.x``, ``.edge_attr``, ``.edge_index`` and ``.batch``,
  and a namespace avoids ``Data.__getattr__``, which is itself a tracing hazard.
- The returned dict is flattened to a tuple in the fixed order of :data:`OUTPUT_NAMES`.

While tracing, the encoder layers are swapped for the explicit stand-ins from
``layers_explicit`` and the two segment reductions for the ScatterND-emitting ones
from ``scatter_ops``. The swap is scoped to this wrapper - the wrapped model is left
untouched, so the same object can still be used for normal PyTorch inference.
"""
import copy
import types
import typing as t

import torch
import torch.nn as nn

from graph_attention_student.torch.export.layers_explicit import explicit_layer_from
from graph_attention_student.torch.export.scatter_ops import segment_max, segment_sum

__all__ = ['MeganOnnxWrapper', 'OUTPUT_NAMES', 'INPUT_NAMES']

#: Fixed output order of the exported graph. Mirrors the keys of the dict returned by
#: ``Megan.forward``; ``node_embedding`` is omitted because it is a large intermediate
#: that no consumer of the exported model has asked for.
OUTPUT_NAMES: t.List[str] = [
    'graph_output',
    'graph_embedding',
    'node_importance',
    'node_importance_norm',
    'edge_importance',
    'edge_importance_norm',
    'graph_variance',
]

#: Fixed input order of the exported graph.
INPUT_NAMES: t.List[str] = [
    'x', 'edge_index', 'edge_attr', 'batch', 'graph_ids', 'node_mask',
]


def _with_dim_size(reduction: t.Callable) -> t.Callable:
    """
    Adapts a ``scatter_ops`` reduction to the signature of ``Megan._segment_max`` and
    ``Megan._segment_sum``, rejecting the ``dim_size=None`` that those allow.

    Inferring the segment count from ``index`` is data-dependent and cannot be traced,
    so every call site in ``Megan.forward`` has to pass ``dim_size`` explicitly.
    """
    def segment(src: torch.Tensor,
                index: torch.Tensor,
                dim_size: t.Optional[int] = None,
                ) -> torch.Tensor:
        if dim_size is None:
            raise ValueError(
                f'{reduction.__name__} was called without dim_size while tracing for '
                f'ONNX export. Every segment reduction in Megan.forward must pass '
                f'dim_size, since the segment count cannot be inferred from the index '
                f'in an exported graph.'
            )
        return reduction(src, index, dim_size)

    return segment


class MeganOnnxWrapper(nn.Module):
    """
    Wraps a :class:`Megan` so that it can be traced by :func:`torch.onnx.export`.

    The wrapper holds a *deep copy* of the model whose encoder layers have been
    replaced by their explicit equivalents, so exporting never mutates the model the
    caller handed in.

    :param model: the trained Megan model. It is put into eval mode on the copy;
        tracing a train-mode model would bake batch statistics into the BatchNorm
        layers that sit on the edge dimension of every V2 attention layer.
    """

    def __init__(self, model: nn.Module):
        super().__init__()

        model_copy = copy.deepcopy(model)
        model_copy.eval()
        model_copy.encoder_layers = nn.ModuleList(
            [explicit_layer_from(lay) for lay in model_copy.encoder_layers]
        )
        # Route the two segment reductions in Megan.forward onto the ScatterND path.
        # Bound as plain functions on the instance, ignoring "self".
        model_copy._segment_max = _with_dim_size(segment_max)
        model_copy._segment_sum = _with_dim_size(segment_sum)

        self.model = model_copy
        self.num_channels = model_copy.num_channels

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor,
                graph_ids: torch.Tensor,
                node_mask: t.Optional[torch.Tensor] = None,
                ) -> t.Tuple[torch.Tensor, ...]:
        """
        :param x: node features, ``(N, node_dim)``
        :param edge_index: COO connectivity, ``(2, E)``, int64
        :param edge_attr: edge features, ``(E, edge_dim)``
        :param batch: graph assignment per node, ``(N,)``, int64
        :param graph_ids: ``arange(num_graphs)``, int64. Only its *length* is read,
            as the number of graphs in the batch. The batch size is communicated as a
            tensor shape rather than as an integer because deriving it in the graph
            (``batch.max() + 1``) is data-dependent and cannot be exported alongside
            a dynamic batch axis, while an integer argument would be baked in as a
            constant and fix the batch size forever.
        :param node_mask: optional ``(N, K)`` channel ablation mask. Passing ones with
            column ``k`` zeroed reproduces one step of ``leave_one_out_deviations``.

        :returns: the tensors named by :data:`OUTPUT_NAMES`, in that order.
        """
        data = types.SimpleNamespace(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
        )
        info = self.model(
            data,
            node_mask=node_mask,
            num_graphs=graph_ids.shape[0],
        )
        return tuple(info[name] for name in OUTPUT_NAMES)
