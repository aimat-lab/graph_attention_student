"""
Construction of ``manifest.json``, the single data contract for a consumer of an
exported model.

Everything a non-Python caller needs in order to run the model *correctly* - rather
than merely run it - goes here: the tensor signature, the hashes, the runtime
versions, and the handful of MEGAN behaviours that are surprising if you have not
read the Python source.
"""
import hashlib
import typing as t

from graph_attention_student.utils import get_version

__all__ = ['sha256_of', 'build_manifest']


def sha256_of(path: str) -> str:
    """
    :returns: the hex SHA-256 of the file at ``path``, read in chunks so that a large
        model does not have to be held in memory twice.
    """
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)

    return digest.hexdigest()


def build_manifest(model_path: str,
                   architecture: dict,
                   inputs: t.List[dict],
                   outputs: t.List[dict],
                   opset: int,
                   external_data: bool,
                   featurization: t.Optional[dict] = None,
                   featurization_reason: t.Optional[str] = None,
                   ) -> dict:
    """
    :param model_path: path of the written ``model.onnx``, hashed into the manifest.
    :param inputs: one ``{'name', 'dtype', 'shape'}`` dict per graph input, where a
        dynamic dimension is given by name rather than by size.
    :param outputs: the same, per graph output.
    :param external_data: whether the model has an external-data sidecar. ONNX Runtime
        Web cannot load one, so this is recorded rather than assumed.
    :param featurization: the featurization spec, or None when it is not portable.
    :param featurization_reason: why the spec was omitted, when it was.
    """
    try:
        import onnxruntime
        onnxruntime_version = onnxruntime.__version__
    except Exception:
        onnxruntime_version = None

    import torch

    return {
        'schemaVersion': 1,
        'generatedBy': f'graph_attention_student {get_version()}',
        'model': {
            'path': 'model.onnx',
            'sha256': sha256_of(model_path),
            'opset': opset,
            'externalData': external_data,
            'torchVersion': torch.__version__,
            'onnxruntimeVersion': onnxruntime_version,
            'inputs': inputs,
            'outputs': outputs,
        },
        'architecture': architecture,
        'featurization': featurization,
        'featurizationOmittedBecause': featurization_reason,
        # The things that make a consumer's output wrong rather than merely absent.
        'notes': {
            'outputActivation': (
                'None. Megan applies no final activation - lay_act_final is an Identity '
                'and is never called in forward - so "graph_output" is raw values for a '
                'regression model and raw LOGITS for a classification model. Apply '
                'softmax or sigmoid yourself.'
            ),
            'leaveOneOut': (
                'Pass "node_mask" as ones of shape (N, num_channels) with column k set '
                'to zero, and subtract that run\'s graph_output from the unmasked run\'s, '
                'to reproduce one channel of leave_one_out_deviations. Run the session '
                'num_channels + 1 times in total.'
            ),
            'graphIds': (
                '"graph_ids" is arange(num_graphs); only its length is read. The batch '
                'size has to arrive as a tensor shape because deriving it inside the '
                'graph would be a data-dependent shape.'
            ),
            'unBatching': (
                'Outputs named graph* are per graph, node* are per node and edge* are '
                'per edge, concatenated over the batch. Split node/edge outputs using '
                '"batch" and "batch[edge_index[0]]" respectively, mirroring '
                'AbstractGraphModel.forward_graphs.'
            ),
            'cudaProviderPrecision': (
                'ONNX Runtime\'s CUDAExecutionProvider uses TF32 matmuls by default on '
                'Ampere and newer, which puts its output about 2e-3 away from the CPU '
                'provider rather than the ~2e-6 you get otherwise. The explanation '
                'masks were unaffected in testing - the same atoms stayed above the '
                'threshold - but pass the provider option use_tf32="0" if you need the '
                'CPU and GPU results to agree tightly. It costs roughly 20% throughput.'
            ),
            'edgelessGraphs': (
                'Molecules with no bonds (e.g. methane) are not supported. This is a '
                'pre-existing limitation of the PyTorch model, not of the export: '
                'MoleculeProcessing emits edge_indices with shape (0,) rather than '
                '(0, 2), and Megan.forward raises IndexError on such a graph.'
            ),
        },
    }
