"""
Trains the MEGAN model on the VGD multitask TADF dataset, which uses all three values "strength",
"splitting" and "absorption" of the original TADF dataset as multitask targets.

**CHANGELOG**

0.1.0 - 20.01.2023 - Initial version
"""
import os
import pathlib

from pycomex.util import Skippable
from pycomex.experiment import SubExperiment

PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH = os.path.expanduser('~/.visual_graph_datasets/datasets/tadf_multi')
TRAIN_RATIO = 0.6
NUM_TARGETS = 3
HAS_IMPORTANCES: bool = False
HAS_GRAPH_ATTRIBUTES: bool = True
EXAMPLES_RATIO: float = 0.01
IMPORTANCE_CHANNELS = 2 * NUM_TARGETS
TARGET_NAMES = [
    'splitting',
    'strength',
    'absorption',
]
REGRESSION_REFERENCE = [
    1,  # splitting
    1,  # strength
    3,  # absorption
]
REGRESSION_LIMITS = [
    [-1, +4],
    [-0, +5],
    [-0, +13],
]

# == TRAINING PARAMETERS ==
BATCH_SIZE = 4096
EPOCHS = 5
LOG_STEP = 1
DEVICE = 'cpu:0'

# == EXPERIMENT PARAMETERS ==
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_multitask_megan.py')
BASE_PATH = os.getcwd()
NAMESPACE = 'results/vgd_multitask_megan_tadf'
DEBUG = True
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):
    pass
