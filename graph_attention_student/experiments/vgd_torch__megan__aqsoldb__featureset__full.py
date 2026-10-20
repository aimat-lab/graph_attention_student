"""
AqSolDB feature-set arm: full.

All 45 precomputed node features, including the five per-atom RDKit descriptors that
RDKit.js cannot produce. This is the reference: what the model achieves when nothing is
withheld.

Everything except the node feature columns is identical to the other arms - same dataset,
same splits, same model, same redesigned explanation objective - so any difference in the
results is attributable to the features alone.
"""
import typing as t

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# :param NODE_FEATURE_COLUMNS:
NODE_FEATURE_COLUMNS: t.Optional[t.List[int]] = None
# :param FEATURE_SET_NAME:
FEATURE_SET_NAME: str = 'full'

__DEBUG__ = False
__TESTING__ = False

experiment = Experiment.extend(
    'vgd_torch__megan__aqsoldb__featureset.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()
