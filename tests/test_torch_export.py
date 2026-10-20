import os
import json
import tempfile

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from torch_geometric.nn.aggr import SumAggregation, MaxAggregation
from visual_graph_datasets.processing.molecules import MoleculeProcessing

from graph_attention_student.torch.megan import Megan
from graph_attention_student.torch.data import data_list_from_graphs
from graph_attention_student.torch.layers import (
    GraphAttentionLayer, GraphAttentionLayerV2, GraphAttentionLayerV3,
    ParallelHeadAttention,
)
from graph_attention_student.torch.export.layers_explicit import (
    explicit_layer_from, UnsupportedLayerError,
)
from graph_attention_student.torch.export.scatter_ops import segment_max, segment_sum
from graph_attention_student.torch.export.featurization import derive_featurization_spec
from graph_attention_student.torch.export.architecture import extract_architecture

onnx = pytest.importorskip('onnx')
ort = pytest.importorskip('onnxruntime')

from graph_attention_student.torch.export.wrapper import (  # noqa: E402
    MeganOnnxWrapper, INPUT_NAMES, OUTPUT_NAMES,
)
from graph_attention_student.torch.export.onnx_export import (  # noqa: E402
    export_megan_onnx, ExportError,
)


SMILES = [
    'CCO', 'c1ccccc1O', 'CC(=O)Oc1ccccc1C(=O)O', 'CN1C=NC2=C1C(=O)N(C)C(=O)N2C',
    'FC(F)(F)c1ccccc1', 'CC(=O)[O-]', 'c1ccsc1', 'C1CC1N',
]

MODEL_KWARGS = dict(
    node_dim=10, edge_dim=4, units=[16, 16], num_channels=2,
    projection_units=[16], final_units=[8, 1],
)


def _graphs(smiles=None):
    """Process SMILES into graph dicts using the default molecule processing."""
    processing = MoleculeProcessing()
    return processing, [processing.process(value) for value in (smiles or SMILES)]


def _model(**overrides):
    """A small, deterministically initialized Megan in eval mode."""
    torch.manual_seed(3)
    kwargs = dict(MODEL_KWARGS)
    kwargs.update(overrides)
    model = Megan(**kwargs)
    model.eval()
    return model


def _tensors(graphs, num_channels=2):
    """The wrapper's input tuple for a list of graphs."""
    data = next(iter(DataLoader(
        data_list_from_graphs(graphs), batch_size=len(graphs), shuffle=False)))
    num_graphs = int(data.batch.max()) + 1
    return data, (
        data.x, data.edge_index, data.edge_attr, data.batch,
        torch.arange(num_graphs, dtype=torch.long),
        torch.ones(data.x.shape[0], num_channels),
    )


# == segment reductions ==

def test_segment_ops_match_pyg():
    """
    ``segment_sum`` and ``segment_max`` should be exactly equal to the PyG reductions
    they replace, including the zero fill that PyG produces for an empty segment.
    """
    torch.manual_seed(0)
    num_segments, features = 7, 5
    src = torch.rand(19, features)
    index = torch.randint(0, num_segments, (19,))
    # Force at least one empty segment, the case where a max reduction could plausibly
    # differ (an -inf fill rather than a zero fill).
    index[index == 3] = 0

    assert torch.equal(
        segment_sum(src, index, num_segments),
        scatter(src, index, dim=0, dim_size=num_segments, reduce='sum'))
    assert torch.equal(
        segment_max(src, index, num_segments),
        scatter(src, index, dim=0, dim_size=num_segments, reduce='max'))


def test_segment_ops_match_the_aggregation_layers():
    """
    The segment reductions should also match the PyG ``Aggregation`` modules that Megan
    uses for pooling, since those are the call sites they replace.
    """
    torch.manual_seed(0)
    num_segments = 4
    src = torch.rand(13, 3)
    index = torch.randint(0, num_segments, (13,))

    assert torch.equal(
        segment_sum(src, index, num_segments),
        SumAggregation()(src, index, dim_size=num_segments))
    assert torch.equal(
        segment_max(src, index, num_segments),
        MaxAggregation()(src, index, dim_size=num_segments))


# == explicit layers ==

@pytest.mark.parametrize('layer_class,kwargs', [
    (GraphAttentionLayer, {}),
    (GraphAttentionLayerV2, {'hidden_dim': 32}),
    (GraphAttentionLayerV3, {'hidden_dim': 32, 'num_heads': 3}),
])
def test_explicit_layers_match_message_passing(layer_class, kwargs):
    """
    Each explicit layer should reproduce its PyG counterpart bit for bit in eval mode.

    This is the test that catches the explicit implementations drifting away from the
    ``message()`` bodies they duplicate.
    """
    torch.manual_seed(1)
    layer = layer_class(16, 16, 4, **kwargs).eval()
    explicit = explicit_layer_from(layer).eval()

    x = torch.rand(9, 16)
    edge_attr = torch.rand(22, 4)
    edge_index = torch.stack([torch.randint(0, 9, (22,)), torch.randint(0, 9, (22,))])

    with torch.no_grad():
        reference = layer(x, edge_attr, edge_index)
        got = explicit(x, edge_attr, edge_index)

    assert torch.equal(reference[0], got[0])
    assert torch.equal(reference[1], got[1])


@pytest.mark.parametrize('num_channels', [1, 2, 4])
def test_explicit_parallel_head_matches(num_channels):
    """
    ``ParallelHeadAttention`` should match for any number of channels, with and without
    the edge weights that the explanation ablation path feeds in.
    """
    torch.manual_seed(1)
    layers = [GraphAttentionLayerV2(16, 16, 4, hidden_dim=32) for _ in range(num_channels)]
    parallel = ParallelHeadAttention(layers, activation='leaky_relu').eval()
    explicit = explicit_layer_from(parallel).eval()

    x = torch.rand(9, 16, num_channels)
    edge_attr = torch.rand(22, 4)
    edge_index = torch.stack([torch.randint(0, 9, (22,)), torch.randint(0, 9, (22,))])

    for edge_weights in (None, torch.rand(22, num_channels)):
        with torch.no_grad():
            reference = parallel(x, edge_attr, edge_index, edge_weights=edge_weights,
                                 return_attention_weights=True)
            got = explicit(x, edge_attr, edge_index, edge_weights=edge_weights)

        assert torch.equal(reference[0], got[0])
        assert torch.equal(reference[1], got[1])


def test_unsupported_layer_is_named():
    """An unknown layer should raise an error that names it, not fail obscurely later."""
    with pytest.raises(UnsupportedLayerError, match='Linear'):
        explicit_layer_from(torch.nn.Linear(2, 2))


# == the wrapper ==

@pytest.mark.parametrize('layer_version', ['v1', 'v2', 'v3'])
def test_wrapper_matches_forward(layer_version):
    """
    The traced wrapper should return exactly what ``Megan.forward`` returns, so that the
    ONNX export is verified against the real inference path rather than a re-derivation
    of it.
    """
    _, graphs = _graphs()
    model = _model(layer_version=layer_version)
    wrapper = MeganOnnxWrapper(model).eval()
    data, args = _tensors(graphs)

    with torch.no_grad():
        reference = model(data)
        got = wrapper(*args)

    for name, value in zip(OUTPUT_NAMES, got):
        assert torch.equal(reference[name], value), f'{name} differs'


def test_wrapper_does_not_mutate_the_model():
    """Exporting must leave the original model usable for normal PyTorch inference."""
    _, graphs = _graphs()
    model = _model()
    data, args = _tensors(graphs)

    with torch.no_grad():
        before = model(data)['graph_output'].clone()
    MeganOnnxWrapper(model)
    with torch.no_grad():
        after = model(data)['graph_output']

    assert torch.equal(before, after)
    assert type(model.encoder_layers[0]).__name__ == 'ParallelHeadAttention'


# == the export ==

@pytest.fixture(scope='module')
def exported():
    """Export a model once and reuse it across the tests that only read the result."""
    processing, graphs = _graphs()
    model = _model()
    directory = tempfile.mkdtemp()
    paths = export_megan_onnx(model, directory, graphs, processing=processing)
    return model, processing, graphs, paths


def test_export_writes_the_expected_artifacts(exported):
    """The export should produce a model, a manifest, fixtures and a featurization spec."""
    _, _, _, paths = exported
    for role in ('model', 'manifest', 'fixtures', 'featurization'):
        assert role in paths, f'no {role} was written'
        assert os.path.getsize(paths[role]) > 0


def test_exported_graph_is_valid_and_uses_only_scatter_nd(exported):
    """
    The graph should pass ``onnx.checker`` and contain no ``ScatterElements``.

    ONNX Runtime Web's WebGPU backend has no ScatterElements kernel, so this is the
    property that keeps the export runnable there.
    """
    _, _, _, paths = exported
    model = onnx.load(paths['model'])
    onnx.checker.check_model(model)

    op_types = {node.op_type for node in model.graph.node}
    assert 'ScatterElements' not in op_types
    assert 'ScatterND' in op_types


def test_export_is_self_contained(exported):
    """
    No external-data sidecar may be produced: ONNX Runtime Web cannot load one and fails
    with "Module.MountedFiles is not available".
    """
    _, _, _, paths = exported
    assert not os.path.exists(paths['model'] + '.data')

    manifest = json.load(open(paths['manifest']))
    assert manifest['model']['externalData'] is False


def test_manifest_is_a_complete_serializable_contract(exported):
    """
    The manifest should describe the model well enough to use it without reading the
    Python source, and must be JSON-serializable.
    """
    _, _, _, paths = exported
    manifest = json.load(open(paths['manifest']))

    assert json.dumps(manifest)
    assert manifest['schemaVersion'] == 1
    assert manifest['model']['opset'] >= 18
    assert len(manifest['model']['sha256']) == 64

    assert [i['name'] for i in manifest['model']['inputs']] == list(INPUT_NAMES)
    assert [o['name'] for o in manifest['model']['outputs']] == list(OUTPUT_NAMES)

    # The dynamic dimensions must be recorded by name, not as the size they happened to
    # have while tracing.
    shapes = {i['name']: i['shape'] for i in manifest['model']['inputs']}
    assert shapes['x'][0] == 'num_nodes'
    assert shapes['edge_index'] == [2, 'num_edges']

    # The behaviours a consumer gets wrong if nobody tells them.
    assert 'logits' in manifest['notes']['outputActivation'].lower()
    assert manifest['notes']['leaveOneOut']
    assert manifest['architecture']['config']['num_channels'] == 2


@pytest.mark.parametrize('subset', [SMILES, SMILES[:1], SMILES[:3], SMILES * 2])
def test_onnx_matches_pytorch_numerically(exported, subset):
    """
    The exported model should reproduce the PyTorch outputs across batch sizes other
    than the one it was traced with, which is what proves the dynamic axes work.
    """
    model, _, _, paths = exported
    wrapper = MeganOnnxWrapper(model).eval()
    processing = MoleculeProcessing()
    graphs = [processing.process(value) for value in subset]
    _, args = _tensors(graphs)

    session = ort.InferenceSession(paths['model'], providers=['CPUExecutionProvider'])
    got = session.run(None, {name: tensor.numpy()
                             for name, tensor in zip(INPUT_NAMES, args)})
    with torch.no_grad():
        reference = wrapper(*args)

    for name, expected, actual in zip(OUTPUT_NAMES, reference, got):
        assert actual.shape == tuple(expected.shape), f'{name} shape differs'
        difference = np.abs(expected.numpy() - actual).max()
        assert difference < 1e-4, f'{name} differs by {difference}'


def test_onnx_preserves_explanations_at_the_task_level(exported):
    """
    Tensor closeness is not enough for the explanations: the normalization applies a
    discontinuous ``where(x < 0.1, x * 0.01, x)``, so a float-level difference can push
    a node across the threshold and visibly change which atoms are highlighted. The
    explanation is only equivalent if the highlighted set and the ranking survive.
    """
    model, _, graphs, paths = exported
    wrapper = MeganOnnxWrapper(model).eval()
    _, args = _tensors(graphs)

    session = ort.InferenceSession(paths['model'], providers=['CPUExecutionProvider'])
    got = dict(zip(OUTPUT_NAMES, session.run(
        None, {name: tensor.numpy() for name, tensor in zip(INPUT_NAMES, args)})))
    with torch.no_grad():
        reference = dict(zip(OUTPUT_NAMES, wrapper(*args)))

    for name in ('node_importance_norm', 'edge_importance_norm'):
        expected = reference[name].numpy()
        actual = got[name]

        # The same elements survive the soft threshold.
        np.testing.assert_array_equal(
            expected >= 0.1, actual >= 0.1,
            err_msg=f'{name}: a different set of elements is above the threshold')

        # And they are ranked the same way within each channel.
        for channel in range(expected.shape[1]):
            np.testing.assert_array_equal(
                np.argsort(-expected[:, channel], kind='stable'),
                np.argsort(-actual[:, channel], kind='stable'),
                err_msg=f'{name}: channel {channel} ranks elements differently')


def test_node_mask_reproduces_leave_one_out(exported):
    """
    Feeding a channel-ablation mask should reproduce ``leave_one_out_deviations``, which
    is how a consumer gets the fidelity values without a K-loop baked into the graph.
    """
    model, _, graphs, paths = exported
    data, args = _tensors(graphs)
    session = ort.InferenceSession(paths['model'], providers=['CPUExecutionProvider'])

    def run(mask):
        feeds = {name: tensor.numpy() for name, tensor in zip(INPUT_NAMES, args)}
        feeds['node_mask'] = mask.numpy()
        return session.run(None, feeds)[OUTPUT_NAMES.index('graph_output')]

    unmasked = run(torch.ones(data.x.shape[0], model.num_channels))
    deviations = np.zeros((len(graphs), model.out_dim, model.num_channels))
    for channel in range(model.num_channels):
        mask = torch.ones(data.x.shape[0], model.num_channels)
        mask[:, channel] = 0.0
        deviations[:, :, channel] = unmasked - run(mask)

    reference = model.leave_one_out_deviations(graphs)
    assert deviations.shape == reference.shape
    assert np.abs(deviations - reference).max() < 1e-4

    # The ordering of the channels by deviation is what an explanation consumer acts on.
    np.testing.assert_array_equal(
        np.argsort(-deviations, axis=-1), np.argsort(-reference, axis=-1))


# == featurization ==

def test_featurization_spec_describes_the_default_processing():
    """
    The derived spec should describe the default molecule processing exactly: where each
    encoder writes, what its vocabulary is, and how wide the result is.
    """
    result = derive_featurization_spec(MoleculeProcessing())
    assert result.portable
    assert result.reason is None

    spec = result.spec
    assert spec['nodeAttributes']['size'] == 10
    assert spec['edgeAttributes']['size'] == 4
    assert spec['doubleEdgesUndirected'] is True

    symbol = spec['nodeAttributes']['encoders'][0]
    assert symbol['property'] == 'GetSymbol'
    assert symbol['kind'] == 'one_hot'
    assert symbol['add_unknown'] is True
    # Eight elements plus the trailing unknown slot.
    assert symbol['slice'] == [0, 9]

    bond = spec['edgeAttributes']['encoders'][0]
    assert bond['property'] == 'GetBondType'
    assert bond['values'] == [1, 2, 3, 12]

    # Must not come from visual_graph_datasets.__version__, which is stale.
    assert spec['visualGraphDatasetsVersion'] is not None
    assert json.dumps(spec)


def test_unportable_encoder_withholds_the_spec_but_still_exports():
    """
    An encoder with no JavaScript equivalent should suppress ``featurization.json`` and
    say why, while the model and manifest export normally. A spec that cannot be
    reproduced is worse than no spec, because it would diverge silently.
    """
    from visual_graph_datasets.processing.molecules import chem_prop, list_identity

    class GasteigerProcessing(MoleculeProcessing):
        node_attribute_map = {
            **MoleculeProcessing.node_attribute_map,
            'chiral_tag': {
                'callback': chem_prop('GetChiralTag', list_identity),
                'description': 'Chiral tag - works fine in Python, but is not one of '
                               'the properties featurizer.js implements',
            },
        }

    processing = GasteigerProcessing()
    result = derive_featurization_spec(processing)
    assert not result.portable
    assert 'GetChiralTag' in result.reason

    # The model and its manifest must still export; only the JS contract is withheld.
    # The extra encoder widens the node features by one, so the model has to match.
    model = _model(node_dim=processing.get_num_node_attributes())
    graphs = [processing.process(value) for value in SMILES[:3]]
    directory = tempfile.mkdtemp()
    paths = export_megan_onnx(model, directory, graphs, processing=processing)

    assert 'featurization' not in paths
    assert not os.path.exists(os.path.join(directory, 'featurization.json'))

    manifest = json.load(open(paths['manifest']))
    assert manifest['featurization'] is None
    assert 'GetChiralTag' in manifest['featurizationOmittedBecause']


# == error handling ==

def test_edgeless_graph_is_rejected_with_an_explanation():
    """
    A molecule without bonds should be refused by name, because Megan itself cannot
    process one - a pre-existing limitation that is confusing when it surfaces as an
    IndexError from inside the export.
    """
    processing, graphs = _graphs(['C'])
    model = _model()
    with pytest.raises(ExportError, match='no edges'):
        export_megan_onnx(model, tempfile.mkdtemp(), graphs, processing=processing)


def test_low_opset_is_rejected():
    """Opset 17 cannot express ScatterND with reduction='max', so it must be refused."""
    processing, graphs = _graphs()
    model = _model()
    with pytest.raises(ExportError, match='opset'):
        export_megan_onnx(model, tempfile.mkdtemp(), graphs,
                          processing=processing, opset=17)


def test_empty_example_graphs_are_rejected():
    """Tracing needs at least one example graph; an empty list must say so."""
    model = _model()
    with pytest.raises(ExportError, match='must not be empty'):
        export_megan_onnx(model, tempfile.mkdtemp(), [])


# == architecture extraction ==

def test_architecture_extraction_is_serializable_and_complete():
    """
    The architecture description is shared with the Rust/WebAssembly compilation path,
    so it has to be JSON-serializable and carry the shape-determining configuration.
    """
    model = _model()
    architecture = extract_architecture(model)

    assert json.dumps(architecture)
    assert architecture['num_parameters_total'] > 0
    assert any(layer['op'] == 'linear' for layer in architecture['layers'])

    config = architecture['config']
    for key in ('units', 'num_channels', 'out_dim', 'prediction_mode',
                'layer_version', 'attention_aggregation'):
        assert key in config, f'{key} missing from the extracted config'
