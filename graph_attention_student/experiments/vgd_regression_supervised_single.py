"""

"""
import os
import pathlib
import typing as t

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from pycomex.experiment import SubExperiment
from pycomex.util import Skippable
from kgcnn.data.utils import ragged_tensor_from_nested_numpy
from kgcnn.layers.conv.gat_conv import AttentionHeadGATV2
from graph_attention_student.models.megan import Megan
from graph_attention_student.models.gradient import grad_importances
from graph_attention_student.models.gnes import GnesGradientModel
from graph_attention_student.training import NoLoss
from graph_attention_student.training import ExplanationLoss
from graph_attention_student.training import mae as mae_loss

PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PROPERTIES ==
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('~/.visual_graph_datasets/datasets/rb_dual_motifs')
NODE_IMPORTANCES_KEY = 'node_importances_1'
EDGE_IMPORTANCES_KEY = 'edge_importances_1'

# == MODEL PARAMETERS ==
SPARSITY_FACTOR: float = 1.0
FINAL_ACTIVATION: str = 'linear'

MEGAN_UNITS: t.List[int] = [12, 12, 12]
MEGAN_FINAL_UNITS: t.List[int] = [6, 1]
MEGAN_IMPORTANCE_CHANNELS: int = 1
MEGAN_IMPORTANCE_FACTOR: float = 0.0

GNES_UNITS: t.List[int] = [12, 12, 12]
GNES_FINAL_UNITS: t.List[int] = [6, 1]
GNES_LAYER_CB: t.Callable = lambda units: AttentionHeadGATV2(
    units=units,
    activation='kgcnn>leaky_relu',
    use_edge_features=True,
)

# == EXPERIMENT PARAMETERS ==
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_regression_supervised.py')
BASE_PATH = PATH
NAMESPACE = 'results/' + os.path.basename(__file__).strip('.py')
DEBUG = True
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):

    @se.hook('model_generator')
    def model_generator(e):
        # ~ fixed GNES version
        # This versions uses the absolute activation, which works better for regression problems
        model = GnesGradientModel(
            units=e.parameters['GNES_UNITS'],
            final_units=e.parameters['GNES_FINAL_UNITS'],
            batch_size=e.parameters['BATCH_SIZE'],
            final_activation=e.parameters['FINAL_ACTIVATION'],
            layer_cb=e.parameters['GNES_LAYER_CB'],
            importance_func=lambda gradient_info: grad_importances(
                gradient_info=gradient_info,
                use_absolute=True,
                use_relu=False,
                keepdims=False
            ),
        )
        model.compile(
            loss=[
                ks.losses.MeanSquaredError(),
                ExplanationLoss(loss_function=mae_loss),
                ExplanationLoss(loss_function=mae_loss),
            ],
            loss_weights=[1, 1, 1],
            metrics=[ks.metrics.MeanSquaredError()],
            optimizer=e.parameters['OPTIMIZER_CB']()
        )
        yield 'gnes_fixed', model

        # ~ original GNES version
        # This versions uses the relu activation, which was proposed in the original paper but which
        # does not work very well for a regression setting.
        model = GnesGradientModel(
            units=e.parameters['GNES_UNITS'],
            final_units=e.parameters['GNES_FINAL_UNITS'],
            batch_size=e.parameters['BATCH_SIZE'],
            final_activation=e.parameters['FINAL_ACTIVATION'],
            layer_cb=e.parameters['GNES_LAYER_CB'],
            importance_func=lambda gradient_info: grad_importances(
                gradient_info=gradient_info,
                use_absolute=False,
                use_relu=True,
                keepdims=False
            ),
        )
        model.compile(
            loss=[
                ks.losses.MeanSquaredError(),
                ExplanationLoss(loss_function=mae_loss),
                ExplanationLoss(loss_function=mae_loss),
            ],
            loss_weights=[1, 1, 1],
            metrics=[ks.metrics.MeanSquaredError()],
            optimizer=e.parameters['OPTIMIZER_CB']()
        )
        yield 'gnes_original', model

        # ~ MEGAN unsupervised
        # This MEGAN version does not receive explanation supervision
        model = Megan(
            units=e.parameters['MEGAN_UNITS'],
            final_units=e.parameters['MEGAN_FINAL_UNITS'],
            importance_channels=e.parameters['MEGAN_IMPORTANCE_CHANNELS'],
            importance_factor=0.0,
            sparsity_factor=e.parameters['SPARSITY_FACTOR'],
            final_activation=e.parameters['FINAL_ACTIVATION'],
            concat_heads=False,
            use_graph_attributes=False
        )
        model.compile(
            loss=[
                ks.losses.MeanSquaredError(),
                NoLoss(),
                NoLoss(),
            ],
            loss_weights=[1, 0, 0],
            metrics=[ks.metrics.MeanSquaredError()],
            optimizer=e.parameters['OPTIMIZER_CB']()
        )
        yield 'megan_1_normal', model

        # ~ MEGAN supervised
        # This MEGAN version does not receive explanation supervision
        model = Megan(
            units=e.parameters['MEGAN_UNITS'],
            final_units=e.parameters['MEGAN_FINAL_UNITS'],
            importance_channels=e.parameters['MEGAN_IMPORTANCE_CHANNELS'],
            importance_factor=0.0,
            sparsity_factor=e.parameters['SPARSITY_FACTOR'],
            final_activation=e.parameters['FINAL_ACTIVATION'],
            concat_heads=False,
            use_graph_attributes=False
        )
        model.compile(
            loss=[
                ks.losses.MeanSquaredError(),
                ExplanationLoss(),
                ExplanationLoss(),
            ],
            loss_weights=[1, 1, 1],
            metrics=[ks.metrics.MeanSquaredError()],
            optimizer=e.parameters['OPTIMIZER_CB']()
        )
        yield 'megan_1_supervised', model

