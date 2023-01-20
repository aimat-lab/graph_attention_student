"""
This example shows how to train a KGCNN GATv2 for multitask graph regression application based on an
existing "visual graph dataset".

Inherits "vgd_multitask_gnn.py"

**CHANGELOG**

0.1.0 - 18.01.2022 - Initial version

0.2.0 - 20.01.2022 - Extended the model class to implement multi-head attention instead of just using single
attention layers.

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
from pycomex.experiment import SubExperiment
from pycomex.util import Skippable
from kgcnn.layers.conv.gat_conv import AttentionHeadGATV2
from kgcnn.layers.modules import LazyConcatenate, DenseEmbedding, DropoutEmbedding, LazyAverage
from kgcnn.layers.pooling import PoolingGlobalEdges

# local
from graph_attention_student.training import mse


# == DATASET PARAMETERS ==

USE_GRAPH_ATTRIBUTES: bool = False
# The GATv2 layers are capable to use edge features. This flag controls if that should be the case or not
# If this flag is False then there won't be any edge information at all but just constant edge weights of
# 1 for every edge.
USE_EDGE_FEATURES: bool = True


# == MODEL PARAMETERS ==

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
            heads = []
            for _ in range(num_heads):
                lay = AttentionHeadGATV2(
                    units=k,
                    activation=activation,
                    use_edge_features=use_edge_features,
                )
                heads.append(lay)

            self.conv_layers.append(heads)

        self.lay_combine = LazyAverage()
        self.lay_combine = LazyConcatenate()
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
        for heads in self.conv_layers:

            xs = []
            for lay in heads:
                _x = lay([x, edge_input, edge_index_input])
                xs.append(_x)

            x = self.lay_combine(xs)
            if training:
                x = self.lay_dropout(x)

        final = self.lay_pooling(x)
        if self.use_graph_attributes:
            final = self.lay_concat([final, graph_input])

        for lay in self.final_layers:
            final = lay(final)

        return final


MODEL_NAME = 'GATv2'
UNITS: t.List[int] = [32, 32, 32]
DROPOUT_RATE: float = 0.25
# The number of attention heads to be used in the network.
NUM_HEADS: int = 3

# == TRAINING PARAMETERS ==
EPOCHS: int = 250
LOG_STEP: int = 5

# == EXPERIMENT PARAMETERS ==
PATH = pathlib.Path(__file__).parent.absolute()
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_multitask_gnn.py')
BASE_PATH = os.getcwd()
NAMESPACE = 'results/vgd_multitask_gnn_gatv2'
DEBUG = True
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):

    # Here we need to overwrite this hook in case we want ot use the edge features. In the original
    # experiment, the implementation of this hook removes the edge features from the dataset, which
    # means by overwriting it we would prevent that and allow the edge features to be part of the dataset
    @se.hook('modify_g', replace=USE_EDGE_FEATURES)
    def allow_edge_features(e, g):
        return g

    @se.hook('create_model')
    def create_model(e):
        e.info('creating the GATv2 model...')
        e.info(f' * UNITS: {e.parameters["UNITS"]}')
        e.info(f' * FINAL_UNITS: {e.parameters["FINAL_UNITS"]}')
        e.info(f' * DROPOUT_RATE: {e.parameters["DROPOUT_RATE"]}')

        model = Gatv2Model(
            units=e.parameters['UNITS'],
            final_units=e.parameters['FINAL_UNITS'],
            num_heads=e.parameters['NUM_HEADS'],
            use_edge_features=e.parameters['USE_EDGE_FEATURES'],
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
