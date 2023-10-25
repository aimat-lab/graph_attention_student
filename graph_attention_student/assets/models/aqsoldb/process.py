import os
import pathlib
import typing as t

import click
import numpy as np
import numpy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
import rdkit
import rdkit.Chem.AllChem
from rdkit import Chem
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import visual_graph_datasets.typing as tc
# processing
from visual_graph_datasets.processing.base import *
from visual_graph_datasets.processing.colors import *
from visual_graph_datasets.processing.molecules import *
# visualization
from visual_graph_datasets.visualization.base import *
from visual_graph_datasets.visualization.colors import *
from visual_graph_datasets.visualization.molecules import *

PATH = pathlib.Path(__file__).parent.absolute()

# -- custom imports --
"""
One way in which this class will be used is by copying its entire source code into a separate
python module, which will then be shipped with each visual graph dataset as a standalone input
processing functionality.

All the code of a class can easily be extracted and copied into a template using the "inspect"
module, but it may need to use external imports which are not present in the template by default.
This is the reason for this method.

Within this method all necessary imports for the class to work properly should be defined. The code
in this method will then be extracted and added to the top of the templated module in the imports
section.
"""
pass
# --


# -- The following class was dynamically inserted --
class VgdMoleculeProcessing(MoleculeProcessing):

    node_attribute_map = {
        'symbol': {
            'callback': chem_prop('GetSymbol', OneHotEncoder(
                ['H', 'C', 'N', 'O', 'B', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'],
                add_unknown=True,
                dtype=str
            )),
            'description': 'one-hot encoding of atom type',
            'is_type': True,
            'encodes_symbol': True,
        },
        'hybridization': {
            'callback': chem_prop('GetHybridization', OneHotEncoder(
                [2, 3, 4, 5, 6],
                add_unknown=True,
                dtype=int,
            )),
            'description': 'one-hot encoding of atom hybridization',
        },
        'total_degree': {
            'callback': chem_prop('GetTotalDegree', OneHotEncoder(
                [0, 1, 2, 3, 4, 5],
                add_unknown=False,
                dtype=int
            )),
            'description': 'one-hot encoding of the degree of the atom'
        },
        'num_hydrogen_atoms': {
            'callback': chem_prop('GetTotalNumHs', OneHotEncoder(
                [0, 1, 2, 3, 4],
                add_unknown=False,
                dtype=int
            )),
            'description': 'one-hot encoding of the total number of attached hydrogen atoms'
        },
        'mass': {
            'callback': chem_prop('GetMass', list_identity),
            'description': 'The mass of the atom'
        },
        'charge': {
            'callback': chem_prop('GetFormalCharge', list_identity),
            'description': 'The charge of the atom',
        },
        'is_aromatic': {
            'callback': chem_prop('GetIsAromatic', list_identity),
            'description': 'Boolean flag of whether the atom is aromatic',
        },
        'is_in_ring': {
            'callback': chem_prop('IsInRing', list_identity),
            'description': 'Boolean flag of whether atom is part of a ring'
        },
        'crippen_contributions': {
            'callback': crippen_contrib(),
            'description': 'The crippen logP contributions of the atom as computed by RDKit'
        },
        'tpsa_contribution': {
            'callback': tpsa_contrib(),
            'description': 'Contribution to TPSA as computed by RDKit',
        },
        'lasa_contribution': {
            'callback': lasa_contrib(),
            'description': 'Contribution to ASA as computed by RDKit'
        },
        'gasteiger_charge': {
            'callback': gasteiger_charges(),
            'description': 'The partial gasteiger charge attributed to atom as computed by RDKit'
        },
        'estate_indices': {
            'callback': estate_indices(),
            'description': 'EState index as computed by RDKit'
        }
    }

    edge_attribute_map = {
        'bond_type': {
            'callback': chem_prop('GetBondType', OneHotEncoder(
                [1, 2, 3, 12],
                add_unknown=False,
                dtype=int,
            )),
            'description': 'one-hot encoding of the bond type',
            'is_type': True,
            'encodes_bond': True,
        },
        'stereo': {
            'callback': chem_prop('GetStereo', OneHotEncoder(
                [0, 1, 2, 3],
                add_unknown=False,
                dtype=int,
            )),
            'description': 'one-hot encoding of the stereo property'
        },
        'is_aromatic': {
            'callback': chem_prop('GetIsAromatic', list_identity),
            'description': 'boolean flag of whether bond is aromatic',
        },
        'is_in_ring': {
            'callback': chem_prop('IsInRing', list_identity),
            'description': 'boolean flag of whether bond is part of ring',
        },
        'is_conjugated': {
            'callback': chem_prop('GetIsConjugated', list_identity),
            'description': 'boolean flag of whether bond is conjugated'
        }
    }

    graph_attribute_map = {
        'molecular_weight': {
            'callback': chem_descriptor(Chem.Descriptors.ExactMolWt, list_identity),
            'description': 'the exact molecular weight of the molecule',
        },
        'num_radical_electrons': {
            'callback': chem_descriptor(Chem.Descriptors.NumRadicalElectrons, list_identity),
            'description': 'the total number of radical electrons in the molecule',
        },
        'num_valence_electrons': {
            'callback': chem_descriptor(Chem.Descriptors.NumValenceElectrons, list_identity),
            'description': 'the total number of valence electrons in the molecule'
        }
    }

# --


# The data element pre-processing capabilities defined in the above class can either be accessed by
# importing this object from this module in other code and using the implementations of the methods
# "process", "visualize" and "create".
# Alternatively this module also acts as a command line tool (see below)
processing = VgdMoleculeProcessing()

if __name__ == '__main__':
    # This class inherits from "click.MultiCommand" which means that it will directly work as a cli
    # entry point when using the __call__ method such as here. This will enable this python module to
    # expose the cli commands defined in the above class when invoking it from the command line.
    processing()
