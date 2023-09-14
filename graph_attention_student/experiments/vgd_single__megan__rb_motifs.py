import os
import pathlib
import typing as t

import tensorflow as tf
import tensorflow.keras as ks
# from pycomex.util import Skippable
# from pycomex.experiment import SubExperiment
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('/media/ssd/.visual_graph_datasets/datasets/rb_dual_motifs')
USE_DATASET_SPLIT: t.Optional[int] = None
TRAIN_RATIO: float = 0.8
NUM_EXAMPLES: int = 100
NUM_TARGETS: int = 1

NODE_IMPORTANCES_KEY: t.Optional[str] = 'node_importances_2'
EDGE_IMPORTANCES_KEY: t.Optional[str] = 'edge_importances_2'

USE_NODE_COORDINATES: bool = False
USE_EDGE_LENGTHS: bool = False
USE_EDGE_ATTRIBUTES: bool = False

# == MODEL PARAMETERS ==
UNITS = [32, 32, 32]
DROPOUT_RATE = 0.0
FINAL_UNITS = [32, 16, 1]
REGRESSION_REFERENCE = -0.0
REGRESSION_WEIGHTS = [1.0, 1.0]
IMPORTANCE_CHANNELS: int = 2
IMPORTANCE_FACTOR = 2.0
IMPORTANCE_MULTIPLIER = 0.5
FIDELITY_FACTOR = 0.2
FIDELITY_FUNCS = [
    lambda org, mod: tf.nn.relu(mod - org),
    lambda org, mod: tf.nn.relu(org - mod),
]
SPARSITY_FACTOR = 0.2
CONCAT_HEADS = False

# == TRAINING PARAMETERS ==
BATCH_SIZE = 16
EPOCHS = 50
REPETITIONS = 1
OPTIMIZER_CB = lambda: ks.optimizers.experimental.AdamW(learning_rate=0.001)

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
