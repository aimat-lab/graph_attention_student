"""
AqSolDB feature-set arm: portable.

The 38 columns reachable from RDKit.js, assuming hybridization gets ported to JavaScript.
Drops crippen (2), tpsa (1), lasa (1), gasteiger (2) and estate (1) - the per-atom
descriptor decompositions that RDKit.js has no API for at all.

Everything except the node feature columns is identical to the other arms - same dataset,
same splits, same model, same redesigned explanation objective - so any difference in the
results is attributable to the features alone.
"""
import typing as t

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# :param NODE_FEATURE_COLUMNS:
NODE_FEATURE_COLUMNS: t.Optional[t.List[int]] = list(range(0, 38))
# :param FEATURE_SET_NAME:
FEATURE_SET_NAME: str = 'portable'

__DEBUG__ = False
__TESTING__ = False

experiment = Experiment.extend(
    'vgd_torch__megan__aqsoldb__featureset.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()
