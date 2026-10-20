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
  with the onnxscript translation in :data:`CUSTOM_TRANSLATION_TABLE`, emitting
  ``ScatterND`` with ``reduction='max'`` (opset 18).

Both are bit-identical to the PyG equivalents for the inputs MEGAN produces. The
``max`` variant relies on the accumulator being zero-initialised, which matches
PyG's behaviour (empty segments come out as zero) as long as the inputs are
non-negative - true throughout MEGAN, where every reduced quantity is a sigmoid
output or a mean of sigmoid outputs.
"""
import torch

__all__ = [
    'segment_sum',
    'segment_max',
    'CUSTOM_TRANSLATION_TABLE',
]


@torch.library.custom_op('megan::segment_max_nd', mutates_args=())
def _segment_max_nd(out: torch.Tensor, index: torch.Tensor,
                    src: torch.Tensor) -> torch.Tensor:
    """
    Scatter-max of ``src`` into the zero-initialized accumulator ``out`` at row
    positions ``index``.

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
    dim_size=dim_size, reduce='max')`` for non-negative ``src``, but lowers to
    ``ScatterND``. Empty segments come out as zero, matching PyG.
    """
    out = src.new_zeros((dim_size, src.shape[-1]))
    return torch.ops.megan.segment_max_nd(out, index, src)


def _build_translation_table() -> dict:
    """
    Build the ``custom_translation_table`` for :func:`torch.onnx.export`.

    Imported lazily so that importing this module does not require onnxscript,
    which is an optional dependency of the ``export`` extra.
    """
    from graph_attention_student.torch.export.onnx_translation import (
        build_translation_table)
    return build_translation_table()


class _LazyTranslationTable(dict):
    """A dict that populates itself from onnxscript on first use."""

    def _ensure(self):
        if not super().__len__():
            self.update(_build_translation_table())

    def __len__(self):
        self._ensure()
        return super().__len__()

    def __iter__(self):
        self._ensure()
        return super().__iter__()

    def keys(self):
        self._ensure()
        return super().keys()

    def items(self):
        self._ensure()
        return super().items()

    def values(self):
        self._ensure()
        return super().values()

    def __getitem__(self, key):
        self._ensure()
        return super().__getitem__(key)


#: Passed as ``custom_translation_table`` to :func:`torch.onnx.export`.
CUSTOM_TRANSLATION_TABLE = _LazyTranslationTable()
