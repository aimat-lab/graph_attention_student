"""
This example contains the code from the "Custom MEGAN Model Training" section of the README.rst 
file of the project. This example illustrates how it is possible to easily train a custom MEGAN model 
using by defining a sub-experiment for an already existing implementation of a training routine.
"""
import os
import typing as t

import tensorflow as tf
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from graph_attention_student.utils import EXPERIMENTS_PATH

# == CUSTOMIZE HERE ==

# -- DATASET CONFIGURATION --
# Fill in the path to your dataset here
VISUAL_GRAPH_DATASET_PATH: str = '/media/ssd/.visual_graph_datasets/datasets/aqsoldb'
# The type of dataset it is
DATASET_TYPE: str = 'regression'  # or 'classification'
# The number of target labels that the dataset has
NUM_TARGETS: int = 1
# the ratio of the dataset to be used for training (rest is test set)
TRAIN_RATIO: float = 0.8
# The number of randomly chosen example elements from the test set to be 
# plotting the explanations for.
NUM_EXAMPLES: int = 100

NODE_IMPORTANCES_KEY: t.Optional[str] = None  # dont modify
EDGE_IMPORTANCES_KEY: t.Optional[str] = None  # dont modify

# -- MODEL CONFIGURATION --
# the numbers of hidden units in the gnn layers
UNITS = [32, 32, 32]
# the number of units in the projection layers
EMBEDDING_UNITS = [32, 64]
# the number of units in the final prediction mlp layers
FINAL_UNITS = [32, NUM_TARGETS]
# Choose the correct activation for regression(linear) vs classification(softmax) 
FINAL_ACTIVATION: str = 'linear'
# Configure the training process
BATCH_SIZE: int = 32
EPOCHS: int = 10
DEVICE: str = 'cpu:0'

# -- EXPLANATION CONFIGURATION --
# The number of distinct explanations to be created
IMPORTANCE_CHANNELS: int = 2
# the weight of the explanation training loss
IMPORTANCE_FACTOR: float = 1.0
# the weight of the fidelity training loss
FIDELITY_FACTOR: float = 0.1
# the weight of the sparsity training loss
SPARSITY_FACTOR: float = 1.0
# the fidelity functionals
FIDELITY_FUNCS = [
    lambda org, mod: tf.nn.relu(mod - org),
    lambda org, mod: tf.nn.relu(org - mod),
]
# Choose "None" in case of classification
REGRESSION_REFERENCE: float = 0.0

# == DO NOT MODIFY ==

__DEBUG__ = False
__TESTING__ = False
experiment = Experiment.extend(
    os.path.join(EXPERIMENTS_PATH, 'vgd_single__megan2.py'),
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

experiment.run_if_main()