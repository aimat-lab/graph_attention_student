"""
AqSolDB feature-set arm: portable, with the molecules kekulized.

Identical to the "portable" arm - the same 38 node columns, the same model, the same
seeded splits - except that every molecule is re-featurized from its SMILES after
Chem.Kekulize(clearAromaticFlags=True). Aromatic rings therefore arrive as alternating
single and double bonds, and nothing is flagged aromatic.

This isolates one question: does the explicit aromatic designation carry signal for the
GNN, over and above what the alternating bond pattern already encodes? Hybridization and
is_conjugated were both measured to be bit-identical with and without kekulization, so
the pi system is still described - only the word "aromatic" is taken away.

Practically it also decides whether a JavaScript featurizer can read RDKit.js bond orders
straight from get_json() and ignore the aromaticity extension block, which is the single
most error-prone part of that port.
"""
import typing as t

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# :param NODE_FEATURE_COLUMNS:
NODE_FEATURE_COLUMNS: t.Optional[t.List[int]] = list(range(0, 38))
# :param FEATURE_SET_NAME:
FEATURE_SET_NAME: str = 'portable_kekulized'
# :param KEKULIZE:
KEKULIZE: bool = True

__DEBUG__ = False
__TESTING__ = False

experiment = Experiment.extend(
    'vgd_torch__megan__aqsoldb__featureset.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()
