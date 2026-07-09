import os
import json
import tempfile

import numpy as np
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.data import VisualGraphDatasetReader

from graph_attention_student.testing import model_from_processing
from graph_attention_student.torch import evaluation as ev

from .util import ARTIFACTS_PATH


SMILES = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'CCCC', 'c1ccncc1', 'CO', 'CCCl', 'CCC', 'CCOC']


def _build_graphs_and_images(output_path: str):
    """Process a handful of SMILES into VGD elements, returning (graphs, image_paths)."""
    processing = MoleculeProcessing()
    graphs, image_paths = [], []
    for i, smiles in enumerate(SMILES):
        processing.create(value=smiles, index=str(i), width=300, height=300, output_path=output_path)
        data = VisualGraphDatasetReader.read_element(path=output_path, name=str(i))
        graph = data['metadata']['graph']
        graph['graph_labels'] = np.array([float(i % 2)], dtype='float32')
        graphs.append(graph)
        image_paths.append(data['image_path'])
    return processing, graphs, image_paths


def test_compute_diagnostics_structure_and_verdicts():
    """
    ``compute_diagnostics`` should return a JSON-serializable report with the five
    axes, the divergence block and an overall verdict - each axis carrying both the
    raw values and a PASS/WARN/FAIL verdict.
    """
    with tempfile.TemporaryDirectory() as path:
        processing, graphs, _ = _build_graphs_and_images(path)
        model = model_from_processing(
            processing=processing, num_outputs=1, num_channels=2, prediction_mode='regression',
            importance_mode='regression', importance_factor=1.0,
        )
        model.eval()

        report = ev.compute_diagnostics(model, graphs, loss_history=[5.0, 3.0, 2.0, 1.8])

        # structure
        assert set(report['axes'].keys()) == {
            'prediction', 'explanation_accuracy', 'fidelity_sign',
            'fidelity_magnitude', 'coverage',
        }
        assert report['overall']['verdict'] in (ev.PASS, ev.WARN, ev.FAIL)
        for axis in report['axes'].values():
            assert axis['verdict'] in (ev.PASS, ev.WARN, ev.FAIL)
        # raw values are present, not just verdicts
        assert 'per_channel_auc' in report['axes']['explanation_accuracy']
        assert len(report['axes']['explanation_accuracy']['per_channel_auc']) == 2
        assert 'per_channel_sign_consistency' in report['axes']['fidelity_sign']

        # JSON serializable
        json.dumps(report)


def test_explanation_accuracy_uncomputable_when_importance_mode_none():
    """
    A model without importance_mode (e.g. reloaded before the hparams fix) cannot
    form the binary proxy target. The axis must report this explicitly
    (computed=False + error) and FAIL, rather than silently faking a ~0.5 AUC that
    looks like a genuinely random model.
    """
    with tempfile.TemporaryDirectory() as path:
        processing, graphs, _ = _build_graphs_and_images(path)
        model = model_from_processing(
            processing=processing, num_outputs=1, num_channels=2, prediction_mode='regression',
            # importance_mode intentionally left as None
        )
        model.eval()

        report = ev.compute_diagnostics(model, graphs)
        ea = report['axes']['explanation_accuracy']
        assert ea['verdict'] == ev.FAIL
        assert ea['computed'] is False
        assert 'error' in ea and 'importance_mode' in ea['error']
        assert ea['per_channel_auc'] is None
        # a broken measurement must not be reported as a random-model mean_auc
        assert ea['mean_auc'] is None
        json.dumps(report)


def test_divergence_detection():
    """Divergence must FAIL on NaN output history and on an exploded explanation loss."""
    with tempfile.TemporaryDirectory() as path:
        processing, graphs, _ = _build_graphs_and_images(path)
        model = model_from_processing(
            processing=processing, num_outputs=1, num_channels=2, prediction_mode='regression',
            importance_mode='regression', importance_factor=1.0,
        )
        model.eval()

        exploded = ev.compute_diagnostics(model, graphs, loss_history=[1.0, 0.5, 0.4, 50.0])
        assert exploded['divergence']['verdict'] == ev.FAIL
        assert exploded['divergence']['expl_loss_exploded'] is True

        nan_run = ev.compute_diagnostics(model, graphs, loss_history=[1.0, float('nan')])
        assert nan_run['divergence']['verdict'] == ev.FAIL


def test_generate_diagnostic_report_writes_artifacts():
    """The full writer should produce report.json plus the three diagnostic images."""
    with tempfile.TemporaryDirectory() as path:
        processing, graphs, image_paths = _build_graphs_and_images(path)
        model = model_from_processing(
            processing=processing, num_outputs=1, num_channels=2, prediction_mode='regression',
            importance_mode='regression', importance_factor=1.0,
        )
        model.eval()

        out = os.path.join(ARTIFACTS_PATH, 'test_diagnostic_report')
        report = ev.generate_diagnostic_report(
            model=model,
            graphs=graphs,
            output_dir=out,
            example_graphs=graphs,
            example_image_paths=image_paths,
            num_examples=6,
        )
        for name in ('report.json', 'fidelity.png', 'examples.png', 'scorecard.png'):
            assert os.path.exists(os.path.join(out, name)), f'missing artifact: {name}'

        # report on disk matches the returned dict
        on_disk = json.load(open(os.path.join(out, 'report.json')))
        assert on_disk['overall']['verdict'] == report['overall']['verdict']
