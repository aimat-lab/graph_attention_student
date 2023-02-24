import os
import pathlib
import typing as t

import tensorflow as tf
import tensorflow.keras as ks
from pycomex.util import Skippable
from pycomex.experiment import SubExperiment
from kgcnn.layers.modules import DenseEmbedding, DropoutEmbedding, LazyConcatenate, LazyAverage
from kgcnn.layers.conv.gat_conv import AttentionHeadGATV2
from kgcnn.layers.pooling import PoolingGlobalEdges

from graph_attention_student.training import mse

# == DATASET PARAMETERS ==
USE_EDGE_ATTRIBUTES = True


# == MODEL PARAMETERS ==
MODEL_NAME = 'GATv2'

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
                 use_edge_features: bool = True):
        super(Gatv2Model, self).__init__()

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
        for lay in self.final_layers:
            final = lay(final)

        return final


MODEL_NAME = 'GATv2'
UNITS: t.List[int] = [32, 32, 32]
DROPOUT_RATE: float = 0.0
# The number of attention heads to be used in the network.
NUM_HEADS: int = 2

# == TRAINING PARAMETERS ==
BATCH_SIZE = 256
REPETITIONS = 5

# == EXPERIMENT PARAMETERS ==
PATH = pathlib.Path(__file__).parent.absolute()
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_single_gnnx_aqsoldb.py')
BASE_PATH = os.getcwd()
NAMESPACE = 'results/vgd_single_gnnx_aqsoldb_gatv2'
DEBUG = True
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):

    @se.hook('create_model', replace=True)
    def create_model(e):
        e.info('GATv2 Model')
        model = Gatv2Model(
            units=e.parameters['UNITS'],
            final_units=e.parameters['FINAL_UNITS'],
            dropout_rate=e.parameters['DROPOUT_RATE'],
        )
        model.compile(
            loss=mse,
            metrics=mse,
            optimizer=e.parameters['OPTIMIZER_CB'](),
            run_eagerly=False
        )
        return model
