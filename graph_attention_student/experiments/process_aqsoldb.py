"""
Processes the AqSolDB dataset for the water solubility of chemical compounds from the raw format as it is
provided by the original authors into a format that can be used by other experiments. The datasets consists
of ~10000 molecules annotated with measurements of water solubility at room temperatures which are to be
predicted. The original format of the dataset is a CSV file of SMILES strings of the molecules. Further
experiments however need it in the format of a so called "eye tracking dataset". In this format the datasets
consists of a folder, where every element is associated with 2 files: (1) An image which contains a
visualization of the corresponding molecule and (2) a json metadata file. The metadata file has to contain
the target solubility value and an appropriate graph structure representing the molecule, which contains
numeric feature vectors for each node and each graph. This processing is done by using RDKit to interpret
and draw the molecules based on the SMILES string and then export the graph representation using some of
atom and bond properties calculated by RDkit.
"""
import os
import sys
import csv
import math
import pathlib
import random
from zipfile import ZipFile
from itertools import chain
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
from kgcnn.data.moleculenet import MoleculeNetDataset, OneHotEncoder

from pycomex.experiment import Experiment
from pycomex.util import Skippable

# from gnn_teacher_student.util import graph_dict_to_list_values
# from gnn_teacher_student.visualization import pdf_from_eye_tracking_dataset
from graph_attention_student.data import create_molecule_eye_tracking_dataset, load_eye_tracking_dataset
from graph_attention_student.util import DATASETS_FOLDER

SHORT_DESCRIPTION = (
    'Processes the raw AqSolDB dataset for water solubility of molucules into a format which can be used by '
    'the other experiments.'
)

PATH = os.path.dirname(pathlib.Path(__file__).parent.absolute())
BASE_PATH = os.getenv('EXPERIMENT_BASE_PATH', os.path.join(PATH, 'experiment_results'))

# == SOURCE PARAMETERS ==
TRAIN_DATASET_PATH = os.path.join(DATASETS_FOLDER, 'aqsoldb_raw', 'dataset-all.csv')
TEST_DATASET_PATH = os.path.join(DATASETS_FOLDER, 'aqsoldb_raw', 'dataset-E.csv')

# == PROCESSING PARAMETERS ==
ATOMS = ['Cl', 'C', 'N', 'O', 'F', 'S', 'Br', 'I', 'H', 'Na', 'P', 'Ca', 'B']
# The dimensions which the images of the molecular graphs are supposed to have in the end, given in pixels
WIDTH, HEIGHT = 900, 900

# == EVALUATION PARAMETERS ==

def plot_distribution(ax: plt.Axes,
                      values: List[float],
                      bins: List[int],
                      color='coral'):

    hist, bin_edges = np.histogram(values, bins)
    xs = list(range(len(hist)))
    ax.bar(xs, hist, color=color)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'[{bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f})' for i in range(len(bin_edges) - 1)])
    ax.set_ylabel('Count')


def molecule_filter(mg, ds):
    data = dict(ds)
    # The paper which acts as the main comparison lists some pre-processing steps which we will recreate
    # here. A compound will not be used if...

    # There is no carbon atom in it at all
    if 'C' not in data['SMILES']:
        return None

    # The compound contains charged atoms
    if '+' in data['SMILES'] or '-' in data['SMILES']:
        return None

    # If the element consists of adjoined mixtures
    if '.' in data['SMILES']:
        return None

    return 1


def load_moleculenet(csv_path: str,
                     name: str = 'aqsoldb',
                     name_blacklist: List[str] = []):
    moleculenet = MoleculeNetDataset(
        file_name=os.path.basename(csv_path),
        data_directory=os.path.dirname(csv_path),
        dataset_name=name
    )

    moleculenet.prepare_data(
        overwrite=False,
        smiles_column_name='SMILES',
        add_hydrogen=True,
        make_conformers=True,
        optimize_conformer=True
    )

    moleculenet.read_in_memory(
        label_column_name=['Solubility'],
        add_hydrogen=False,
        has_conformers=True
    )

    moleculenet.set_attributes(
        nodes=['Symbol', 'TotalNumHs'],
        encoder_nodes={
            'Symbol': OneHotEncoder(ATOMS, dtype='str', add_unknown=False),
            'TotalNumHs': int
        },
        edges=['BondType'],
        encoder_edges={
            'BondType': int
        },
        graph=['NumAtoms'],
        additional_callbacks={
            'node_count': lambda mg, ds: int(mg.mol.GetNumAtoms()),
            'solubility': lambda mg, ds: float(dict(ds)['Solubility']),
            'smiles': lambda mg, ds: str(dict(ds)['SMILES']),
            'id': lambda mg, ds: str(dict(ds)['ID']),
            'name': lambda mg, ds: str(dict(ds)['Name']),
            # AqSolDB consists of multiple datasets, which are enumerated with characters A, B, C, D...
            # which are encoded into the ID of each element and here we retrieve it, because that will
            # later be important to determine the test set.
            'origin': lambda mg, ds: str(dict(ds)['ID']).split('-')[0].upper(),
            # This is a boolean value which is None if we do not want to include that compound into the
            # dataset.
            'filter': molecule_filter,
            'name_filter': lambda mg, ds: None if str(dict(ds)['Name']) in name_blacklist else 1
        }
    )
    moleculenet.clean(['filter', 'name_filter', 'edge_attributes'])
    return moleculenet


NAMESPACE = 'process_aqsoldb'
BASE_PATH = os.getcwd()
DEBUG = True

with Skippable(), (e := Experiment(base_path=BASE_PATH, namespace=NAMESPACE, glob=globals())):

    # -- Loading the original dataset from CSV
    e.info('Loading CSV file...')
    # For each line in the CSV a dict will be added to this list
    raw_dataset: List[dict] = []
    with open(TRAIN_DATASET_PATH, mode='r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',', quotechar='"')
        for row in reader:
            raw_dataset.append(row)

    e.info(f'{len(raw_dataset)} rows of data')
    e.info(f'{len(reader._fieldnames)} fields: {", ".join(reader._fieldnames)}')
    e.info(f'printing one example:')
    for key, value in raw_dataset[0].items():
        e.info(f' * {key:<20}: {value}')

    # -- Processing the dataset with KGCNN
    test_moleculenet = load_moleculenet(TEST_DATASET_PATH)
    e.info(f'loaded test dataset with {len(test_moleculenet)} elements')

    test_names = [g['name'] for g in test_moleculenet]
    train_moleculenet = load_moleculenet(TRAIN_DATASET_PATH, name_blacklist=test_names)
    e.info(f'loaded train dataset with {len(train_moleculenet)} elements')

    # -- Actually creating the eye tracking dataset
    dataset_path = os.path.join(e.path, 'dataset')
    os.mkdir(dataset_path)

    infos = {}
    for dataset_type, dataset in [('train', train_moleculenet), ('test', test_moleculenet)]:
        for g in dataset:
            infos[str(g['id'])] = {
                'smiles': str(g['smiles']),
                'id': str(g['id']),
                'input_type': 'regression',
                'solubility': float(g['solubility']),
                'value': float(g['solubility']),
                'name': str(g['name']),
                'type': dataset_type
            }

    create_molecule_eye_tracking_dataset(
        molecule_infos=infos,
        dest_path=dataset_path,
        image_width=WIDTH,
        image_height=HEIGHT,
        logger=e.logger,
        set_attributes_kwargs={
            'nodes': ['Symbol', 'TotalDegree', 'FormalCharge', 'NumRadicalElectrons', 'Hybridization',
                      'IsAromatic', 'IsInRing', 'TotalNumHs', 'CIPCode', 'ChiralityPossible', 'ChiralTag'],
            'encoder_nodes': {
                'Symbol': OneHotEncoder(
                    ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'],
                    dtype="str"
                ),
                'Hybridization': OneHotEncoder([2, 3, 4, 5, 6]),
                'TotalDegree': OneHotEncoder([0, 1, 2, 3, 4, 5], add_unknown=False),
                'TotalNumHs': OneHotEncoder([0, 1, 2, 3, 4], add_unknown=False),
                'CIPCode': OneHotEncoder(['R', 'S'], add_unknown=False, dtype='str'),
                'ChiralityPossible': OneHotEncoder(["1"], add_unknown=False, dtype='str'),
            },
            'edges': ['BondType', 'IsAromatic', 'IsConjugated', 'IsInRing', 'Stereo'],
            'encoder_edges': {
                'BondType': OneHotEncoder([1, 2, 3, 12], add_unknown=False),
                'Stereo': OneHotEncoder([0, 1, 2, 3], add_unknown=False)
            }
        }
    )

with Skippable(), e.analysis:
    # The previous create command only creates the dataset files, here we load it into memory as a dict
    # structure so that we can create the overview pdf from this.
    # "pdf_from_eye_tracking_dataset" creates a PDF file which contains some general information about the
    # dataset as well as all the images and the importance ground truth illustrations!
    eye_tracking_dataset = load_eye_tracking_dataset(dataset_path)
    pass




