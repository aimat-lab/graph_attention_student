import pytest
import os
import json
import tempfile

from kgcnn.data.moleculenet import MoleculeNetDataset

from graph_attention_student.data import eye_tracking_dataset_from_moleculenet_dataset

from .util import ASSETS_PATH, LOG


@pytest.fixture()
def moleculenet():
    # This CSV contains 30 molecules as smiles
    csv_path = os.path.join(ASSETS_PATH, 'test.csv')
    moleculenet = MoleculeNetDataset(
        file_name=os.path.basename(csv_path),
        data_directory=os.path.dirname(csv_path),
        dataset_name='temp',
        verbose=0
    )
    moleculenet.prepare_data(
        overwrite=True,
        smiles_column_name='smiles',
        add_hydrogen=True,
        make_conformers=False,
        optimize_conformer=False
    )
    moleculenet.read_in_memory(
        label_column_name='value',
        add_hydrogen=False,
        has_conformers=False
    )
    return moleculenet


def test_eye_tracking_dataset_from_moleculenet_dataset_basically_works(moleculenet):
    assert isinstance(moleculenet, MoleculeNetDataset)
    assert len(moleculenet) != 0

    with tempfile.TemporaryDirectory() as path:
        eye_tracking_dataset = eye_tracking_dataset_from_moleculenet_dataset(
            moleculenet=moleculenet,
            dest_path=path,
            # This should also work, essentially means we just use the default
            set_attributes_kwargs={},
            logger=LOG,
        )

        # First of all we check the dict that is returned by the function
        assert isinstance(eye_tracking_dataset, dict)
        assert len(eye_tracking_dataset) == 30

        # And then if the dataset was even saved as persistent files
        files = os.listdir(path)
        assert len(files) != 0
        assert len(files) == 30 * 2

        for file_name in files:
            name, extension = file_name.split('.')
            file_path = os.path.join(path, file_name)
            # Check if the json files are not empty and contain the most important keys
            if extension in ['json']:
                with open(file_path) as file:
                    content = file.read()
                    data = json.loads(content)
                    assert isinstance(data, dict)
                    assert len(data) != 0
                    assert 'index' in data
                    assert 'name' in data
                    assert 'graph' in data
