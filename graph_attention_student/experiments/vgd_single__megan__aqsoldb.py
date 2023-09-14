import os
import pathlib
import typing as t

import tensorflow as tf
import tensorflow.keras as ks
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('/media/ssd/.visual_graph_datasets/datasets/aqsoldb')
USE_DATASET_SPLIT: t.Optional[int] = 0
TRAIN_RATIO: float = 0.8
NUM_EXAMPLES: int = 100
NUM_TARGETS: int = 1

NODE_IMPORTANCES_KEY: t.Optional[str] = None
EDGE_IMPORTANCES_KEY: t.Optional[str] = None

USE_NODE_COORDINATES: bool = False
USE_EDGE_LENGTHS: bool = False
USE_EDGE_ATTRIBUTES: bool = True

# == MODEL PARAMETERS ==
UNITS = [64, 64, 64]
DROPOUT_RATE = 0.0
FINAL_UNITS = [64, 32, 16, 1]
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
SPARSITY_FACTOR = 0.2
CONCAT_HEADS = False

# == TRAINING PARAMETERS ==
BATCH_SIZE = 8
EPOCHS = 50
REPETITIONS = 1
OPTIMIZER_CB = lambda: ks.optimizers.experimental.AdamW(
    learning_rate=0.001,
    weight_decay=0.01,
)
DEVICE: str = 'cpu:0'

# == EXPERIMENT PARAMETERS ==
__DEBUG__ = True
__TESTING__ = True

experiment = Experiment.extend(
    'vgd_single__megan.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

experiment.run_if_main()
