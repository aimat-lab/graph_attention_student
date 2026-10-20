"""
ONNX-friendly segment reductions.

The MEGAN forward pass uses exactly two kinds of segment reduction - ``add`` and
``max`` - and both go through :func:`torch_geometric.utils.scatter` or a PyG
``Aggregation`` module. Neither exports usefully: the dynamo exporter lowers them
to ``ScatterElements``, and ONNX Runtime Web's **WebGPU execution provider has no
ScatterElements kernel at all**. Only ``ScatterND`` is available there.

This module provides the two reductions in a form that lowers to ``ScatterND``:

- ``segment_sum`` uses ``index_add_``, which the exporter already lowers to
  ``ScatterND`` with ``reduction='add'``.
- ``segment_max`` has no such aten spelling, so it is a custom operator paired
  with the onnxscript translation in ``onnx_translation``, emitting
  ``ScatterND`` with ``reduction='max'`` (opset 18).

Both are bit-identical to the PyG equivalents for any input. The ``max`` variant
accumulates into a ``-inf``-filled tensor and then zeroes the segments that stayed
at ``-inf``, which reproduces PyG exactly: the true maximum for a non-empty segment,
even an all-negative one, and zero for an empty segment.
"""
import torch

__all__ = [
    'segment_sum',
    'segment_max',
]


@torch.library.custom_op('megan::segment_max_nd', mutates_args=())
def _segment_max_nd(out: torch.Tensor, index: torch.Tensor,
                    src: torch.Tensor) -> torch.Tensor:
    """
    Scatter-max of ``src`` into the accumulator ``out`` at row positions ``index``.

    ``out`` is an *input* rather than a shape argument on purpose: it gives the
    ONNX translation a real tensor to take the output shape from, so the
    onnxscript function never has to materialize a symbolic shape.
    """
    idx = index.unsqueeze(-1).expand(-1, src.shape[-1])
    return out.scatter_reduce(0, idx, src, reduce='amax', include_self=True)


@_segment_max_nd.register_fake
def _(out, index, src):
    return torch.empty_like(out)


def segment_sum(src: torch.Tensor, index: torch.Tensor,
                dim_size: int) -> torch.Tensor:
    """
    Sum the rows of ``src`` into ``dim_size`` segments given by ``index``.

    Equivalent to ``torch_geometric.utils.scatter(src, index, dim=0,
    dim_size=dim_size, reduce='sum')``, but lowers to ``ScatterND``.
    """
    out = src.new_zeros((dim_size, src.shape[-1]))
    return out.index_add_(0, index, src)


def segment_max(src: torch.Tensor, index: torch.Tensor,
                dim_size: int) -> torch.Tensor:
    """
    Max-reduce the rows of ``src`` into ``dim_size`` segments given by ``index``.

    Equivalent to ``torch_geometric.utils.scatter(src, index, dim=0,
    dim_size=dim_size, reduce='max')``, but lowers to ``ScatterND``. Empty segments
    come out as zero, matching PyG.
    """
    # A zero-filled accumulator would clamp every segment max to >= 0, so start from
    # -inf and zero only the segments that no row reached.
    out = src.new_full((dim_size, src.shape[-1]), float('-inf'))
    out = torch.ops.megan.segment_max_nd(out, index, src)
    return torch.where(torch.isneginf(out), torch.zeros_like(out), out)

