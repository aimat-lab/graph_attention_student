"""
This experiment will train a baseline GNN model on a visual graph dataset and then will produce 
the explanations using the GNNExplainer (GNNX) method afterwards.
"""
import os
import pathlib
import random
import typing as t
from collections import Counter

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
from pycomex.util import Skippable
from pycomex.experiment import SubExperiment
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.visualization.importances import create_importances_pdf
from kgcnn.data.utils import ragged_tensor_from_nested_numpy
from kgcnn.layers.modules import DropoutEmbedding, DenseEmbedding, LazyConcatenate
from kgcnn.layers.conv.gcn_conv import GCN
from kgcnn.layers.pooling import PoolingGlobalEdges

import graph_attention_student.typing as tc
from graph_attention_student.data import process_graph_dataset
from graph_attention_student.models import gnnx_importances
from graph_attention_student.training import mse, NoLoss
from graph_attention_student.training import LogProgressCallback
from graph_attention_student.util import array_normalize, binary_threshold

# == DATASET PARAMETERS ==
# These parameters define the dataset that is to be used for the experiment as well as properties of
# that dataset such as the train test split for example.

# :param IMPORTANCE_CHANNELS: 
#       The number of importance channels reflected in the ground truth importances loaded from the dataset!
#       The dataset has perhaps
IMPORTANCE_CHANNELS = 1
# :param NODE_IMPORTANCES_KEY:
#       The string key within the graph dict that is used to store the ground truth node importances array.
#       Note that providing providing ground truth importances is optional for a VGD dataset. Setting this 
#       variable to None will disable the comparison with the ground truth importances.
#       Also note that if a dataset contains the information about the ground truth importances, it is also 
#       possible to have different ones (for a different number of distinct explanations / channels) which 
#       may be indicated by a suffix to the string key!
NODE_IMPORTANCES_KEY: t.Optional[str] = 'node_importances_1'
# :param EDGE_IMPORTANCES_KEY:
#       The string key within the graph dict that is used to store the ground truth edge importances array.
EDGE_IMPORTANCES_KEY: t.Optional[str] = 'edge_importances_1'
# :param USE_NODE_COORDINATES:
# If this flag is true, the "node_coordinates" will be added to the other node attributes
USE_NODE_COORDINATES: bool = False

# If this flag is true, the "edge_lengths" will be added to the other edge attributes
USE_EDGE_LENGTHS: bool = False

# In this default implementation we use a GCN network, which cannot handle edge attributes!
USE_EDGE_ATTRIBUTES: bool = False

# == MODEL PARAMETERS ==

MODEL_NAME = 'GNNX+GCN'

class GcnModel(ks.models.Model):

    def __init__(self,
                 units: t.List[int],
                 final_units: t.List[int],
                 pooling_method: str = 'sum',
                 activation: str = 'kgcnn>leaky_relu',
                 final_pooling: str = 'sum',
                 final_activation: str = 'linear',
                 dropout_rate: float = 0.0):
        super(GcnModel, self).__init__()

        self.lay_dropout = DropoutEmbedding(rate=dropout_rate)
        self.conv_layers = []
        for k in units:
            lay = GCN(units=k, pooling_method=pooling_method, activation=activation)
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
        node_input, edge_input, edge_index_input = inputs

        x = node_input
        for lay in self.conv_layers:
            x = lay([x, edge_input, edge_index_input])
            if training:
                x = self.lay_dropout(x)

        final = self.lay_pooling(x)
        for lay in self.final_layers:
            final = lay(final)

        return final


UNITS = [32, 32, 32]
FINAL_UNITS = [32, 16, 1]
DROPOUT_RATE = 0.0

# == TRAINING PARAMETERS ==
# These parameters control the training process of the neural network.

# The number of independent training process repetitions to get a statistical measure of the performance
REPETITIONS: int = 1
# This optimizer will be used during training
OPTIMIZER: ks.optimizers.Optimizer = ks.optimizers.Nadam(learning_rate=0.01)
BATCH_SIZE: int = 256
EPOCHS: int = 250
DEVICE: str = 'gpu:0'

# == GNNX PARAMETERS ==
# Parameters for the GnnExplainer training to create the explanations

GNNX_EPOCHS = 500
GNNX_LEARNING_RATE = 0.02
GNNX_NODE_SPARSITY = 0.3
GNNX_EDGE_SPARSITY = 0.3

# == EVALUATION PARAMETERS ==
# These parameters control the evaluation process, which includes the drawing of visualizations and plots

# After how many elements a log step is printed to the console
LOG_STEP_EVAL: int = 100
# This is the batch size that is used during the evaluation of the test set.
BATCH_SIZE_EVAL: int = 256

# == EXPERIMENT PARAMETERS ==
PATH = pathlib.Path(__file__).parent.absolute()
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_single.py')
BASE_PATH = os.getcwd()
NAMESPACE = 'results/vgd_gnnx'
DEBUG = True
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):

    @se.hook('create_model')
    def create_model(e):
        model = GcnModel(
            units=e.parameters['UNITS'],
            final_units=e.parameters['FINAL_UNITS'],
            dropout_rate=e.parameters['DROPOUT_RATE']
        )
        model.compile(
            loss=mse,
            metrics=mse,
            optimizer=e.parameters['OPTIMIZER_CB'](),
            run_eagerly=False
        )
        return model

    @se.hook('fit_model')
    def fit_model(e, model, x_train, y_train, x_test, y_test):
        history = model.fit(
            x_train,
            y_train,
            epochs=e.parameters['EPOCHS'],
            batch_size=e.parameters['BATCH_SIZE'],
            validation_data=(x_test, y_test),
            validation_freq=1,
            callbacks=[
                LogProgressCallback(
                    e.logger,
                    epoch_step=5,
                    identifier='val_mean_squared_error'
                )
            ],
            verbose=0
        )
        return history.history

    @se.hook('query_model')
    def query_model(e, model, x, y, include_importances: bool = True):
        e.info('querying the model...')
        out_pred = model(x)

        if include_importances:
            e.info('creating GNNX explanations...')
            ni_gnnx, ei_gnnx = gnnx_importances(
                model,
                x=x,
                y=out_pred,
                node_sparsity_factor=e.parameters['GNNX_NODE_SPARSITY'],
                edge_sparsity_factor=e.parameters['GNNX_EDGE_SPARSITY'],
                epochs=e.parameters['GNNX_EPOCHS'],
                learning_rate=e.parameters['GNNX_LEARNING_RATE'],
                logger=e.logger
            )
            return out_pred, ni_gnnx, ei_gnnx

        else:
            return out_pred

    @se.hook('calculate_fidelity')
    def calculate_fidelity(e, model, indices_true, x_true, y_true, out_pred, ni_pred, ei_pred):
        node_masks = []
        edge_masks = []
        for c, (ni, ei) in enumerate(zip(ni_pred, ei_pred)):
            node_mask = np.ones_like(ni)
            node_mask[array_normalize(ni) > 0.5] = 0
            node_masks.append(node_mask)

            edge_mask = np.ones_like(ei)
            edge_mask[array_normalize(ei) > 0.5] = 0
            edge_masks.append(edge_mask)

        node_masks_tensor = ragged_tensor_from_nested_numpy(node_masks)
        edge_masks_tensor = ragged_tensor_from_nested_numpy(edge_masks)

        x_masked = (
            x_true[0] * node_masks_tensor,
            x_true[1] * edge_masks_tensor,
            *x_true[2:]
        )
        out_masked = model(x_masked)
        out_masked = out_masked.numpy()

        fidelities = []
        for c, (value_pred, value_masked) in enumerate(zip(out_pred, out_masked)):
            fidelity = np.abs(value_pred - value_masked)
            fidelities.append(fidelity)

        return fidelities

