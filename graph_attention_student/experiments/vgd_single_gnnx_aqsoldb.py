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
EXAMPLE_INDICES = [7691, 7535, 7175, 4672, 8025, 4735, 7794, 7750]
IMPORTANCE_CHANNELS = 1
NODE_IMPORTANCES_KEY: t.Optional[str] = None
EDGE_IMPORTANCES_KEY: t.Optional[str] = None

USE_NODE_COORDINATES: bool = False
USE_EDGE_LENGTHS: bool = False

# == MODEL PARAMETERS ==
UNITS = [32, 32, 32]
FINAL_UNITS = [32, 16, 1]

# == TRAINING PARAMETERS ==
REPETITIONS = 5
EPOCHS = 250

# == EXPERIMENT PARAMETERS ==
PATH = pathlib.Path(__file__).parent.absolute()
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_single_gnnx.py')
BASE_PATH = os.getcwd()
NAMESPACE = 'results/vgd_single_gnnx_aqsoldb'
DEBUG = True
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):
    pass