"""
The canonical ONNX export for MEGAN models.

:func:`export_megan_onnx` writes a directory containing the model, the metadata a
non-Python consumer needs to feed it correctly, and fixtures that pin the expected
outputs so an independent implementation can verify itself.
"""
import json
import os
import typing as t

import numpy as np
import torch
from torch.export import Dim

from graph_attention_student.torch.export.architecture import extract_architecture
from graph_attention_student.torch.export.featurization import derive_featurization_spec
from graph_attention_student.torch.export.manifest import build_manifest
from graph_attention_student.torch.export.scatter_ops import CUSTOM_TRANSLATION_TABLE
from graph_attention_student.torch.export.wrapper import (
    MeganOnnxWrapper, INPUT_NAMES, OUTPUT_NAMES)

__all__ = ['export_megan_onnx', 'ExportError']

#: Opset 18 is the floor: ScatterND only gained reduction='max' there, and the segment
#: max is unavoidable in MEGAN (the edge-to-node importance pooling is a max even at
#: batch size one).
DEFAULT_OPSET: int = 18

#: Generous finite bounds for the dynamic dimensions. They have to be finite - an
#: unbounded Dim defaults to INT64_MAX and torch.export then fails on a guard
#: asserting the size is not INT64_MAX.
MAX_NODES: int = 2 ** 20
MAX_EDGES: int = 2 ** 20
MAX_GRAPHS: int = 2 ** 16


class ExportError(RuntimeError):
    """Raised when a model or its example data cannot be exported."""


def _validate_graphs(graphs: t.List[dict]) -> None:
    """
    Rejects example graphs the model cannot process, with a message that says why.

    :raises ExportError: on a graph with no edges.
    """
    for index, graph in enumerate(graphs):
        edges = np.asarray(graph['edge_indices'])
        if edges.ndim != 2 or edges.shape[0] == 0:
            raise ExportError(
                f'Example graph {index} has no edges (edge_indices has shape '
                f'{edges.shape}). Megan cannot process an edgeless graph - this is a '
                f'pre-existing limitation of the PyTorch model, which raises IndexError '
                f'on such a graph, and not something the export introduces. Molecules '
                f'like methane ("C") hit it. Please use example graphs with at least '
                f'one bond.'
            )


def _tensors_from_graphs(graphs: t.List[dict], num_channels: int) -> t.Tuple[torch.Tensor, ...]:
    """
    Builds the wrapper's input tuple from a list of graph dicts, going through the same
    ``data_list_from_graphs`` + PyG collation the PyTorch inference path uses.
    """
    from torch_geometric.loader import DataLoader
    from graph_attention_student.torch.data import data_list_from_graphs

    data = next(iter(DataLoader(
        data_list_from_graphs(graphs), batch_size=len(graphs), shuffle=False)))
    num_graphs = int(data.batch.max()) + 1

    return (
        data.x,
        data.edge_index,
        data.edge_attr,
        data.batch,
        torch.arange(num_graphs, dtype=torch.long),
        torch.ones(data.x.shape[0], num_channels),
    )


def _io_description(tensors: t.Sequence[torch.Tensor],
                    names: t.Sequence[str],
                    dynamic: t.Dict[str, t.Dict[int, str]],
                    ) -> t.List[dict]:
    """
    Describes each tensor for the manifest, naming dynamic dimensions instead of
    recording the (meaningless) size they happened to have at trace time.
    """
    described = []
    for name, tensor in zip(names, tensors):
        axes = dynamic.get(name, {})
        shape = []
        for position, size in enumerate(tensor.shape):
            axis = axes.get(position)
            if axis is None:
                shape.append(int(size))
            else:
                # A torch.export.Dim is not JSON-serializable, so record its name.
                shape.append(getattr(axis, '__name__', str(axis)))
        described.append({
            'name': name,
            'dtype': 'int64' if tensor.dtype == torch.long else 'float32',
            'shape': shape,
        })

    return described


def export_megan_onnx(model,
                      out_dir: str,
                      example_graphs: t.List[dict],
                      processing=None,
                      opset: int = DEFAULT_OPSET,
                      ) -> dict:
    """
    Exports a trained MEGAN model to ONNX, together with its metadata and fixtures.

    The model is not modified: the wrapper traces a deep copy whose encoder layers have
    been swapped for ONNX-exportable equivalents.

    :param model: the trained :class:`Megan` instance.
    :param out_dir: directory to write into; created if it does not exist.
    :param example_graphs: graph dicts used both to trace the model and to generate the
        verification fixtures. Use a handful of representative, differently-sized graphs.
    :param processing: the :class:`MoleculeProcessing` used to build the training data.
        When given, its featurization contract is derived and written to
        ``featurization.json``, provided every encoder has a JavaScript equivalent.
    :param opset: ONNX opset. Must be at least 18.

    :raises ExportError: on an unsupported model or unusable example graphs.

    :returns: a dict of the paths written, keyed by role.
    """
    if opset < DEFAULT_OPSET:
        raise ExportError(
            f'opset {opset} is too low: the segment max lowers to ScatterND with '
            f'reduction="max", which requires opset {DEFAULT_OPSET} or higher.'
        )

    if hasattr(model, 'models'):
        raise ExportError(
            'MeganEnsemble cannot be exported. Export each member model separately and '
            'combine their outputs on the consumer side.'
        )

    if not example_graphs:
        raise ExportError('example_graphs must not be empty; it is needed to trace the model.')

    _validate_graphs(example_graphs)
    os.makedirs(out_dir, exist_ok=True)

    wrapper = MeganOnnxWrapper(model).eval()
    args = _tensors_from_graphs(example_graphs, wrapper.num_channels)

    node_dim = Dim('num_nodes', min=1, max=MAX_NODES)
    edge_dim = Dim('num_edges', min=1, max=MAX_EDGES)
    graph_dim = Dim('num_graphs', min=1, max=MAX_GRAPHS)
    dynamic_shapes = {
        'x': {0: node_dim},
        'edge_index': {1: edge_dim},
        'edge_attr': {0: edge_dim},
        'batch': {0: node_dim},
        'graph_ids': {0: graph_dim},
        'node_mask': {0: node_dim},
    }

    model_path = os.path.join(out_dir, 'model.onnx')
    try:
        torch.onnx.export(
            wrapper,
            args,
            model_path,
            dynamo=True,
            opset_version=opset,
            # A sidecar .data file cannot be loaded by ONNX Runtime Web, which fails with
            # "Module.MountedFiles is not available". MEGAN models are small, so a
            # self-contained file is always the right trade here.
            external_data=False,
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            dynamic_shapes=dynamic_shapes,
            custom_translation_table=CUSTOM_TRANSLATION_TABLE,
        )
    except Exception as exc:
        raise ExportError(f'torch.onnx.export failed for this model: {exc}') from exc

    _assert_scatter_nd_only(model_path)

    # ~ fixtures
    # Written from the PyTorch pipeline so that an independent implementation has
    # something to check itself against without needing Python.
    with torch.no_grad():
        expected = wrapper(*args)

    fixtures = {
        'fixtureVersion': 1,
        'inputs': {
            name: {
                'data': tensor.numpy().ravel().tolist(),
                'dims': list(tensor.shape),
                'type': 'int64' if tensor.dtype == torch.long else 'float32',
            }
            for name, tensor in zip(INPUT_NAMES, args)
        },
        'expected': {
            name: {'data': tensor.numpy().ravel().tolist(), 'dims': list(tensor.shape)}
            for name, tensor in zip(OUTPUT_NAMES, expected)
        },
    }
    if processing is not None:
        # Per-molecule featurization fixtures, so an independent featurizer can check
        # itself column by column without having to run the model at all. This is what
        # catches the subtle divergences - a wrong one-hot slot, or RDKit.js reporting
        # kekulized bond orders where Python reports aromatic ones.
        fixtures['featurization'] = [
            {
                'smiles': graph.get('graph_repr'),
                'node_attributes': np.asarray(graph['node_attributes']).tolist(),
                'edge_indices': np.asarray(graph['edge_indices']).tolist(),
                'edge_attributes': np.asarray(graph['edge_attributes']).tolist(),
            }
            for graph in example_graphs
        ]

    written = {'model': model_path}

    fixtures_path = os.path.join(out_dir, 'fixtures.json')
    with open(fixtures_path, 'w') as file:
        json.dump(fixtures, file)
    written['fixtures'] = fixtures_path

    # ~ featurization
    featurization_spec, featurization_reason = None, None
    if processing is not None:
        result = derive_featurization_spec(processing)
        if result.portable:
            featurization_spec = result.spec
            path = os.path.join(out_dir, 'featurization.json')
            with open(path, 'w') as file:
                json.dump(result.spec, file, indent=2)
            written['featurization'] = path
        else:
            featurization_reason = result.reason

    # ~ manifest
    manifest = build_manifest(
        model_path=model_path,
        architecture=extract_architecture(model),
        inputs=_io_description(args, INPUT_NAMES, dynamic_shapes),
        outputs=_io_description(expected, OUTPUT_NAMES, {
            name: {0: 'num_graphs' if name.startswith('graph') else
                   ('num_edges' if name.startswith('edge') else 'num_nodes')}
            for name in OUTPUT_NAMES
        }),
        opset=opset,
        external_data=False,
        featurization=featurization_spec,
        featurization_reason=featurization_reason,
    )
    manifest_path = os.path.join(out_dir, 'manifest.json')
    with open(manifest_path, 'w') as file:
        json.dump(manifest, file, indent=2)
    written['manifest'] = manifest_path

    return written


def _assert_scatter_nd_only(model_path: str) -> None:
    """
    Fails the export if the graph contains ``ScatterElements``.

    ONNX Runtime Web's WebGPU backend has no ScatterElements kernel, so a graph
    containing one silently loses WebGPU support. Catching it here turns a
    hard-to-diagnose runtime failure in someone's browser into an export-time error.
    """
    import onnx

    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    offenders = sorted({node.op_type for node in model.graph.node
                        if node.op_type == 'ScatterElements'})
    if offenders:
        raise ExportError(
            f'The exported graph contains {offenders}, which ONNX Runtime Web\'s WebGPU '
            f'backend cannot execute. Every segment reduction must go through '
            f'graph_attention_student.torch.export.scatter_ops, which emits ScatterND.'
        )
