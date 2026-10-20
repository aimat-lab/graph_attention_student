"""
The molecule processing that a browser can actually reproduce.

This is the production ``VgdMoleculeProcessing`` with the five per-atom RDKit descriptors
removed - Crippen, TPSA, Labute ASA, Gasteiger and EState. Those are decompositions of a
molecule-level quantity onto atoms, and RDKit.js exposes no per-atom API at all, so they
are unreachable in a browser regardless of effort.

Dropping them was measured, not assumed: across five paired seeds on AqSolDB the resulting
model was statistically indistinguishable from the full 45-column one on test R2
(delta = -0.009, p = 0.18), at a small but consistent cost in explanation AUC
(delta = -0.024, p = 0.03). Hybridization was deliberately KEPT, because dropping it as
well cost four times as much accuracy (delta = -0.036, p = 0.035) - and it turned out to
be reproducible in JavaScript after all.

Every property here has a verified equivalent in ``megan.js``, next to this module.
"""
from rdkit import Chem
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.processing.molecules import chem_prop, chem_descriptor
from visual_graph_datasets.processing.base import OneHotEncoder
from visual_graph_datasets.processing.base import list_identity

__all__ = ['PortableMoleculeProcessing']


class PortableMoleculeProcessing(MoleculeProcessing):
    """
    38 node attributes and 11 edge attributes, all reproducible from RDKit.js.

    The column order is identical to the first 38 columns of the production processing, so
    a model trained on ``NODE_FEATURE_COLUMNS = range(0, 38)`` of the existing dataset
    consumes exactly what this produces.
    """

    node_attribute_map = {
        'symbol': {
            'callback': chem_prop('GetSymbol', OneHotEncoder(
                ['H', 'C', 'N', 'O', 'B', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br',
                 'Te', 'I', 'At'],
                add_unknown=True,
                dtype=str,
            )),
            'description': 'one-hot encoding of atom type',
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
                dtype=int,
            )),
            'description': 'one-hot encoding of the degree of the atom',
        },
        'num_hydrogen_atoms': {
            'callback': chem_prop('GetTotalNumHs', OneHotEncoder(
                [0, 1, 2, 3, 4],
                add_unknown=False,
                dtype=int,
            )),
            'description': 'one-hot encoding of the total number of attached hydrogen atoms',
        },
        'mass': {
            'callback': chem_prop('GetMass', list_identity),
            'description': 'The mass of the atom',
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
            'description': 'Boolean flag of whether atom is part of a ring',
        },
    }

    edge_attribute_map = {
        'bond_type': {
            'callback': chem_prop('GetBondType', OneHotEncoder(
                [1, 2, 3, 12],
                add_unknown=False,
                dtype=int,
            )),
            'description': 'one-hot encoding of the bond type',
        },
        'stereo': {
            'callback': chem_prop('GetStereo', OneHotEncoder(
                [0, 1, 2, 3],
                add_unknown=False,
                dtype=int,
            )),
            'description': 'one-hot encoding of the stereo property',
        },
        'is_aromatic': {
            'callback': chem_prop('GetIsAromatic', list_identity),
            'description': 'boolean flag of whether bond is aromatic',
        },
        'is_in_ring': {
            'callback': chem_prop('IsInRing', list_identity),
            'description': 'boolean flag of whether bond is part of a ring',
        },
        'is_conjugated': {
            'callback': chem_prop('GetIsConjugated', list_identity),
            'description': 'boolean flag of whether bond is conjugated',
        },
    }

    graph_attribute_map = {
        'molecular_weight': {
            'callback': chem_descriptor(Chem.Descriptors.ExactMolWt, list_identity),
            'description': 'the exact molecular weight of the molecule',
        },
    }


processing = PortableMoleculeProcessing()
