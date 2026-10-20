"""
AqSolDB feature-set arm: portable_strict.

The 32 columns that are safely implementable in JavaScript today. Drops the five per-atom
descriptors AND hybridization, because RDKit derives hybridization with a perception
algorithm that accounts for conjugation - an amide nitrogen is SP2, not SP3 - so a naive
JavaScript derivation from degree and bond order would silently disagree.

Everything except the node feature columns is identical to the other arms - same dataset,
same splits, same model, same redesigned explanation objective - so any difference in the
results is attributable to the features alone.
"""
import typing as t

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# :param NODE_FEATURE_COLUMNS:
NODE_FEATURE_COLUMNS: t.Optional[t.List[int]] = list(range(0, 17)) + list(range(23, 38))
# :param FEATURE_SET_NAME:
FEATURE_SET_NAME: str = 'portable_strict'

__DEBUG__ = False
__TESTING__ = False

experiment = Experiment.extend(
    'vgd_torch__megan__aqsoldb__featureset.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()
