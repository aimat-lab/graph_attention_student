"""
Coverage for the Python API code blocks documented in README.rst.

Mirrors the two snippets under "Python API" and "Loading and Using Models":
training via :class:`SmilesDataset` + :class:`Megan` + ``pl.Trainer``, then
``Megan.load`` + ``forward_graph`` + ``megan_prediction_report``. Sizes and
epoch counts are scaled down so the tests run quickly; the public API
surface exercised is identical to what the README promises.
"""
import os
import pytest
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from visual_graph_datasets.processing.molecules import MoleculeProcessing

from graph_attention_student import Megan, SmilesDataset
from graph_attention_student.torch.advanced import megan_prediction_report

from .util import ASSETS_PATH


README_CSV = os.path.join(ASSETS_PATH, 'readme_fixture.csv')


@pytest.fixture(scope='module')
def processing() -> MoleculeProcessing:
    return MoleculeProcessing()


@pytest.fixture(scope='module')
def trained_checkpoint(tmp_path_factory, processing) -> str:
    """
    Trains the README "Python API" training snippet end-to-end on the tiny
    fixture CSV and returns the saved checkpoint path. Scoped to the module
    so the two README tests share one trained model.
    """
    tmp_path = tmp_path_factory.mktemp('readme_training')

    dataset = SmilesDataset(
        dataset=README_CSV,
        smiles_column='smiles',
        target_columns=['target'],
        processing=processing,
    )
    loader = DataLoader(dataset, batch_size=4)

    model = Megan(
        node_dim=processing.get_num_node_attributes(),
        edge_dim=processing.get_num_edge_attributes(),
        units=[16, 16],
        final_units=[16, 1],
        prediction_mode='regression',
        importance_factor=1.0,
        importance_mode='regression',
    )

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='cpu',
        default_root_dir=str(tmp_path),
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=loader)
    model.eval()

    ckpt_path = str(tmp_path / 'model.ckpt')
    model.save(ckpt_path)
    return ckpt_path


def test_readme_python_api_training(trained_checkpoint):
    """
    README "Python API" training snippet: SmilesDataset + DataLoader +
    Megan + pl.Trainer.fit + model.save produces a checkpoint file.
    """
    assert os.path.exists(trained_checkpoint)
    assert os.path.getsize(trained_checkpoint) > 0


def test_readme_python_api_load_infer_report(trained_checkpoint, processing, tmp_path):
    """
    README "Loading and Using Models" snippet: Megan.load, forward_graph
    on a SMILES string, and megan_prediction_report writing a PDF.
    """
    model = Megan.load(trained_checkpoint)
    model.eval()

    results = model.forward_graph(processing.process("CCO"))
    assert 'graph_output' in results
    assert results['graph_output'].shape == (1,)

    # save/load round-trip: predictions on the same input must match.
    reloaded = Megan.load(trained_checkpoint)
    reloaded.eval()
    again = reloaded.forward_graph(processing.process("CCO"))
    assert results['graph_output'].item() == pytest.approx(again['graph_output'].item())

    report_path = tmp_path / "report.pdf"
    megan_prediction_report(
        value="CCO",
        model=model,
        processing=processing,
        output_path=str(report_path),
    )
    assert report_path.exists()
    assert report_path.stat().st_size > 0
