import os
import pathlib
import typing as t
import random

import tensorflow as tf
import tensorflow.keras as ks
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

PATH: str = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==

VISUAL_GRAPH_DATASET_PATH: str = 'aqsoldb'
TEST_INDICES_PATH = os.path.join(PATH, 'assets', 'test_indices__logp.json')
USE_DATASET_SPLIT: t.Optional[int] = 0
TRAIN_RATIO: float = 0.8
NUM_EXAMPLES: int = 100
NUM_TEST: int = 1000
NUM_TARGETS: int = 1

NODE_IMPORTANCES_KEY: t.Optional[str] = None
EDGE_IMPORTANCES_KEY: t.Optional[str] = None

USE_NODE_COORDINATES: bool = False
USE_EDGE_LENGTHS: bool = False
USE_EDGE_ATTRIBUTES: bool = True

# == MODEL PARAMETERS ==
UNITS = [32, 32, 32]
DROPOUT_RATE = 0.0
EMBEDDING_UNITS = [64, 32, 16]
FINAL_UNITS = [16, 1]
REGRESSION_REFERENCE = -3.0
REGRESSION_WEIGHTS = [1.0, 1.0]
IMPORTANCE_CHANNELS: int = 2
IMPORTANCE_FACTOR = 2.0
IMPORTANCE_MULTIPLIER = 0.5
FIDELITY_FACTOR = 0.1
FIDELITY_FUNCS = [
    lambda org, mod: tf.nn.relu(mod - org),
    lambda org, mod: tf.nn.relu(org - mod),
]
SPARSITY_FACTOR: float = 1.0
CONCAT_HEADS: bool = False
FINAL_DROPOUT: float = 0.0

# == TRAINING PARAMETERS ==
BATCH_SIZE = 32
EPOCHS = 100
REPETITIONS = 1
OPTIMIZER_CB = lambda: ks.optimizers.experimental.AdamW(
    learning_rate=0.001,
    weight_decay=0.01,
)
DEVICE: str = 'cpu:0'

# == EXPERIMENT PARAMETERS ==
__DEBUG__ = True
__TESTING__ = False

experiment = Experiment.extend(
    'vgd_single__megan2.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('filter_indices')
def training_bootstrap(e: Experiment,
                       train_indices: t.List[int],
                       test_indices: t.List[int],
                       rep: int,
                       **kwargs,
                       ):
    """
    This hook is being called right after the train / test split is created in the main experiment. The function 
    receives the lists of integer train and test indices and is supposed to return a tuple (train_indices, test_indices).
    
    This specific hook implementation realizes training "bootstrapping" - in this process the training indices are 
    randomly sampled from themselves such that randomly some are removed and some appear multiple times. The purpose 
    of this is to create slightly different training distributions for the training of a model ensemble for example.
    """

    e.log('applying bootstrapping by subsampling the training indices...')

    # random.choices() will randomly sample from the given list but is "drawing with replacement" which means 
    # that elements are no longer unique and can appear multiple times. This also means that it is possible 
    # that some elements do not appear at all.
    train_indices = random.choices(train_indices, k=len(train_indices))

    return train_indices, test_indices

experiment.run_if_main()


