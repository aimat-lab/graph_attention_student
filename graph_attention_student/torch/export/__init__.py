"""
Export of trained MEGAN models to portable formats.

The entry point is :func:`export_megan_onnx`, which writes an ONNX model plus the
metadata a non-Python consumer needs in order to feed it correctly.
"""
__all__ = ['export_megan_onnx']


def __getattr__(name):
    # Imported lazily so that "import graph_attention_student.torch.export.wrapper"
    # does not pull in onnx / onnxscript, which are optional dependencies.
    if name == 'export_megan_onnx':
        from graph_attention_student.torch.export.onnx_export import export_megan_onnx
        return export_megan_onnx
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
