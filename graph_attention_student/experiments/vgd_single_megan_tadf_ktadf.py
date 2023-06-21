import os
import pathlib
import typing as t

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
# from pycomex.util import Skippable
# from pycomex.experiment import SubExperiment
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('/media/ssd/.visual_graph_datasets/datasets/tadf')
USE_DATASET_SPLIT: t.Optional[int] = None
TRAIN_RATIO: float = 0.99
NUM_EXAMPLES: int = 100
NUM_TARGETS: int = 1

NODE_IMPORTANCES_KEY: t.Optional[str] = None
EDGE_IMPORTANCES_KEY: t.Optional[str] = None

USE_NODE_COORDINATES: bool = False
USE_EDGE_LENGTHS: bool = False
USE_EDGE_ATTRIBUTES: bool = True

# == MODEL PARAMETERS ==
UNITS = [64, 64, 64, 64]
DROPOUT_RATE = 0.1
FINAL_UNITS = [64, 32, 16, 1]
REGRESSION_REFERENCE = [[-22.]]
REGRESSION_WEIGHTS = [[1.0, 1.0]]
IMPORTANCE_CHANNELS: int = 2
IMPORTANCE_FACTOR = 2.0
IMPORTANCE_MULTIPLIER = 0.8
FIDELITY_FACTOR = 0.2
FIDELITY_FUNCS = [
    lambda org, mod: tf.nn.relu(mod - org),
    lambda org, mod: tf.nn.relu(org - mod),
]
SPARSITY_FACTOR = 0.1
CONCAT_HEADS = False

# == TRAINING PARAMETERS ==
DEVICE = 'cpu:0'
BATCH_SIZE = 32
EPOCHS = 25
REPETITIONS = 1
OPTIMIZER_CB = lambda: ks.optimizers.experimental.AdamW(learning_rate=0.001)

# == EXPERIMENT PARAMETERS ==
LOG_STEP = 10_000
__DEBUG__ = True
__TESTING__ = False

experiment = Experiment.extend(
    'vgd_single_megan.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


@experiment.hook('get_graph_labels')
def get_graph_labels(e, metadata, graph):
    value = float(metadata['target'][2])
    if value < 1e-30:
        raise ValueError()
    return np.log10(np.expand_dims(np.array(value), axis=0))


experiment.run_if_main()