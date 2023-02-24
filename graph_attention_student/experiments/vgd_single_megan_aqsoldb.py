import os
import pathlib
import typing as t

from pycomex.util import Skippable
from pycomex.experiment import SubExperiment

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('~/.visual_graph_datasets/datasets/aqsoldb')
USE_DATASET_SPLIT: t.Optional[int] = 0
TRAIN_RATIO: float = 0.8
NUM_EXAMPLES: int = 100

NODE_IMPORTANCES_KEY: t.Optional[str] = None
EDGE_IMPORTANCES_KEY: t.Optional[str] = None

USE_NODE_COORDINATES: bool = False
USE_EDGE_ATTRIBUTES: bool = True
USE_EDGE_LENGTHS: bool = False

# == MODEL PARAMETERS ==
UNITS = [32, 32, 32]
FINAL_UNITS = [64, 32, 1]
REGRESSION_REFERENCE = -3
REGRESSION_LIMITS = [[-12, 3]]

# == TRAINING PARAMETERS ==
BATCH_SIZE = 128
REPETITIONS = 5

# == EXPERIMENT PARAMETERS ==
PATH = pathlib.Path(__file__).parent.absolute()
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_single_megan.py')
BASE_PATH = PATH
NAMESPACE = 'results/vgd_single_megan_aqsoldb'
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):
    pass