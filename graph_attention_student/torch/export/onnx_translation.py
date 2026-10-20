"""
The onnxscript translation for the custom segment-max operator.

Kept in its own module because :func:`onnxscript.script` compiles the decorated
function's source and resolves its free names against *module* globals, so the
function cannot be defined inside another function. Importing this module requires
onnxscript; ``scatter_ops`` therefore imports it lazily, and stays usable for plain
PyTorch inference on installations without the ``export`` extra.
"""
import torch
from onnxscript import script, opset18 as op
from onnxscript import FLOAT, INT64

__all__ = ['onnx_segment_max_nd', 'build_translation_table']


@script()
def onnx_segment_max_nd(out: FLOAT[...], index: INT64[...],
                        src: FLOAT[...]) -> FLOAT[...]:
    """
    ``ScatterND`` with ``reduction='max'`` - the segment max, as one ONNX node.

    ``out`` is the zero-initialized accumulator and supplies the output shape, so
    no symbolic shape has to be constructed here. ``index`` is ``(E,)`` and becomes
    ``(E, 1)``, which makes ScatterND address whole rows of ``out``.
    """
    indices = op.Unsqueeze(index, [-1])
    return op.ScatterND(out, indices, src, reduction='max')


def build_translation_table() -> dict:
    """
    :returns: the ``custom_translation_table`` mapping for :func:`torch.onnx.export`.
    """
    return {torch.ops.megan.segment_max_nd.default: onnx_segment_max_nd}
