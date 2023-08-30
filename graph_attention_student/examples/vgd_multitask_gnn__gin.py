"""
This example shows how to train a KGCNN GIN model for multitask graph regression application based on an
existing "visual graph dataset".

Inherits "vgd_multitask_gnn.py"

**CHANGELOG**

0.1.0 - 18.01.2022 - Initial version
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
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from kgcnn.layers.conv.gat_conv import AttentionHeadGATV2
from kgcnn.layers.mlp import GraphMLP
from kgcnn.layers.conv.gin_conv import GIN
from kgcnn.layers.modules import LazyConcatenate, DenseEmbedding, DropoutEmbedding
from kgcnn.layers.pooling import PoolingGlobalEdges

# local
from graph_attention_student.training import mse


# == DATASET PARAMETERS ==

USE_GRAPH_ATTRIBUTES: bool = True
USE_EDGE_FEATURES: bool = False


# == MODEL PARAMETERS ==

class GinModel(ks.models.Model):

    def __init__(self,
                 units: t.List[t.List[int]],
                 final_units: t.List[int],
                 pooling_method: str = 'sum',
                 activation: str = 'kgcnn>leaky_relu',
                 final_pooling: str = 'sum',
                 final_activation: str = 'linear',
                 dropout_rate: float = 0.0,
                 use_graph_attributes: bool = False):
        super(GinModel, self).__init__()
        self.use_graph_attributes = use_graph_attributes

        self.lay_dropout = DropoutEmbedding(rate=dropout_rate)
        self.conv_layers: t.List[t.Tuple[ks.layers.Layer, ks.layers.Layer]] = []
        for k in units:
            lay = GIN(
                pooling_method=pooling_method,
                epsilon_learnable=True,
            )
            mlp = GraphMLP(
                units=k,
                activation=activation,
                use_dropout=True,
                rate=dropout_rate,
            )
            self.conv_layers.append((lay, mlp))

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
        for lay, mlp in self.conv_layers:
            x = lay([x, edge_index_input])
            x = mlp(x)
            if training:
                x = self.lay_dropout(x)

        final = self.lay_pooling(x)
        if self.use_graph_attributes:
            final = self.lay_concat([final, graph_input])

        for lay in self.final_layers:
            final = lay(final)

        return final


MODEL_NAME = 'GIN'
DROPOUT_RATE = 0.1
UNITS = [[32, 32], [32, 32], [32, 32]]

# == TRAINING PARAMETERS ==
LEARNING_RATE = 0.001
EPOCHS: int = 1000

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

@experiment.hook('create_model', replace=True)
def create_model(e):
    e.log('creating the GIN model...')
    e.log(f' * UNITS: {e.parameters["UNITS"]}')
    e.log(f' * FINAL_UNITS: {e.parameters["FINAL_UNITS"]}')
    e.log(f' * DROPOUT_RATE: {e.parameters["DROPOUT_RATE"]}')

    model = GinModel(
        units=e.parameters['UNITS'],
        final_units=e.parameters['FINAL_UNITS'],
        use_graph_attributes=e.parameters['USE_GRAPH_ATTRIBUTES'],
        dropout_rate=e.parameters['DROPOUT_RATE']
    )

    model.compile(
        loss=mse,
        metrics=mse,
        optimizer=ks.optimizers.Adam(learning_rate=e.parameters['LEARNING_RATE']),
        run_eagerly=False
    )

    return model


experiment.run_if_main()

