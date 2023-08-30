"""
This example shows how to train a KGCNN GATv2 for multitask graph regression application based on an
existing "visual graph dataset".
"""
import os
import warnings
import pathlib
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import typing as t

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as ks
# from pycomex.experiment import SubExperiment
# from pycomex.util import Skippable
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from kgcnn.layers.conv.gat_conv import AttentionHeadGATV2
from kgcnn.layers.modules import LazyConcatenate, DenseEmbedding, DropoutEmbedding, LazyAverage
from kgcnn.layers.pooling import PoolingGlobalEdges

# local
from graph_attention_student.training import mse
from graph_attention_student.layers import MultiHeadGATV2Layer


# == DATASET PARAMETERS ==
# These parameters are used to specify the dataset to be used for the training as well as additional
# properties of the dataset such as the train test split for example.

# The name of the visual graph dataset to use for this experiment.
VISUAL_GRAPH_DATASET_PATH = os.path.expanduser('~/.visual_graph_datasets/datasets/organic_solvents')
# The ratio of how many elements of the dataset are supposed to be used for the training dataset.
# The rest of them will be used for the test set.
TRAIN_RATIO = 0.8
# The number of target values for each graph in the dataset.
NUM_TARGETS = 4
# Whether the dataset already contains importance (explanation) ground truth annotations.
# most of the time this will most likely not be the case
HAS_IMPORTANCES: bool = False
# The ratio of the test set to be used as examples for the visualization of the explanations
EXAMPLES_RATIO: float = 0.2
# The string names of the target values in the order in which they appear in the dataset as well
# which will be used in the labels for the result visualizations
TARGET_NAMES: t.List[str] = [
    'water',
    'benzene',
    'acetone',
    'ethanol',
]

# IF the dataset includes global graph attributes and if they are supposed to be used in the training
# process, this flag has to be set to True.
USE_GRAPH_ATTRIBTUES: bool = False
# The GATv2 layers are capable to use edge features. This flag controls if that should be the case or not
# If this flag is False then there won't be any edge information at all but just constant edge weights of
# 1 for every edge.
USE_EDGE_FEATURES: bool = True


# == MODEL PARAMETERS ==
# All the parameters that are related to the configuration of the model itself.

# This is the name of the model that will be used in the titles of the created evaluation artifacts and the 
# log messages.
MODEL_NAME = 'GATv2 fix'
# :param UNITS: A list of integers where each integer defines the number of hidden units in one if the 
#       message passing layers of the model.Adding ore elements to this list will configure the model with
#       more layers.
UNITS: t.List[int] = [32, 32, 32]
# :param FINAL_UNITS: A list of integers where each integer defines the number of hidden units in one of 
#       final dense layers of the predicton network. Adding more elements to this list will configure that 
#       network to have more layers.
FINAL_UNITS: t.List[int] = [32, 16]
# :param DROPOUT_RATE: The dropout rate that is being applied during training to the node embeddings after 
#       each message passing layer
DROPOUT_RATE: float = 0.0
# The number of attention heads to be used in the network.
NUM_HEADS: int = 2


class Gatv2Model(ks.models.Model):

    def __init__(self,
                 units: t.List[int],
                 final_units: t.List[int],
                 pooling_method: str = 'sum',
                 num_heads: int = 2,
                 activation: str = 'kgcnn>leaky_relu',
                 final_pooling: str = 'sum',
                 final_activation: str = 'linear',
                 dropout_rate: float = 0.0,
                 use_graph_attributes: bool = False,
                 use_edge_features: bool = True):
        super(Gatv2Model, self).__init__()
        self.use_graph_attributes = use_graph_attributes

        self.lay_dropout = DropoutEmbedding(rate=dropout_rate)
        self.conv_layers = []
        for k in units:

            lay = MultiHeadGATV2Layer(
                units=k,
                num_heads=num_heads,
                activation=activation,
                use_edge_features=use_edge_features,
                concat_heads=False,
                concat_self=True,
            )
            self.conv_layers.append(lay)

        self.lay_pooling = PoolingGlobalEdges(pooling_method=final_pooling)
        self.lay_concat = LazyConcatenate(axis=-1)

        self.final_layers = []
        self.final_activations = ['relu' for _ in final_units]
        self.final_activations[-1] = final_activation
        for k, act in zip(final_units, self.final_activations):
            lay = DenseEmbedding(units=k, activation=act)
            self.final_layers.append(lay)

    def call(self, inputs, training=False):
        if self.use_graph_attributes:
            node_input, edge_input, edge_index_input, graph_input = inputs
        else:
            node_input, edge_input, edge_index_input = inputs

        x = node_input
        for lay in self.conv_layers:

            x, _ = lay([x, edge_input, edge_index_input])

            if training:
                x = self.lay_dropout(x)

        final = self.lay_pooling(x)
        if self.use_graph_attributes:
            final = self.lay_concat([final, graph_input])

        for lay in self.final_layers:
            final = lay(final)

        return final


# == TRAINING PARAMETERS ==
# The parameters that control the training process

# :param REPETITONS: The number of independent repetitions to repeat the entire model training and evaluation 
#       process for.
REPETITIONS: int = 1
# :oaram EPOCHS: The number of epochs to train the model for 
EPOCHS: int = 200
# :param BATCH_SIZE: The nubmer of elements from the training set used during the loss calculation and 
#       subsequent update of the 
BATCH_SIZE: int = 64
# :param EPOCH_STEP: The number of epochs after which to log the progress
EPOCH_STEP: int = 10

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'vgd_multitask_gnn.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

# Here we need to overwrite this hook in case we want ot use the edge features. In the original
# experiment, the implementation of this hook removes the edge features from the dataset, which
# means by overwriting it we would prevent that and allow the edge features to be part of the dataset
@experiment.hook('modify_g', replace=USE_EDGE_FEATURES)
def allow_edge_features(e, g):
    return g

@experiment.hook('create_model')
def create_model(e):
    e.log(f'creating the {e.MODEL_NAME} model...')
    e.log(f' * UNITS: {e.UNITS}')
    e.log(f' * FINAL_UNITS: {e.FINAL_UNITS}')
    e.log(f' * DROPOUT_RATE: {e.DROPOUT_RATE}')

    model = Gatv2Model(
        units=e.UNITS,
        final_units=e.FINAL_UNITS,
        num_heads=e.NUM_HEADS,
        use_edge_features=e.USE_EDGE_FEATURES,
        use_graph_attributes=e.USE_GRAPH_ATTRIBUTES,
        dropout_rate=e.DROPOUT_RATE,
    )

    model.compile(
        loss=mse,
        metrics=mse,
        optimizer=ks.optimizers.Adam(learning_rate=e.LEARNING_RATE),
        run_eagerly=False
    )

    return model


experiment.run_if_main()