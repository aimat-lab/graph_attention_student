import os
import pathlib
import typing as t

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from pycomex.experiment import SubExperiment
from pycomex.util import Skippable
from kgcnn.data.utils import ragged_tensor_from_nested_numpy
from graph_attention_student.models import Megan
from graph_attention_student.training import NoLoss
from graph_attention_student.training import ExplanationLoss

PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PROPERTIES ==
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('~/.visual_graph_datasets/datasets/rb_dual_motifs')
NODE_IMPORTANCES_KEY = 'node_importances_2'
EDGE_IMPORTANCES_KEY = 'edge_importances_2'

# == MODEL PARAMETERS ==
UNITS: t.List[int] = [6, 6, 6]
FINAL_UNITS: t.List[int] = [3, 1]
FINAL_ACTIVATION: str = 'linear'
IMPORTANCE_CHANNELS: int = 2
IMPORTANCE_FACTOR: float = 1.0
IMPORTANCE_MULTIPLIER: float = 2.0
REGRESSION_LIMITS = [[-3, +3]]
REGRESSION_REFERENCE = [0]
CHANNEL_DIRECTIONS = {0: -1, 1: 1}


# == EXPERIMENT PARAMETERS ==
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_regression_supervised.py')
BASE_PATH = PATH
NAMESPACE = 'results/' + os.path.basename(__file__).strip('.py')
DEBUG = True
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):

    @se.hook('model_generator')
    def model_generator(e):
        model = Megan(
            units=UNITS,
            importance_channels=IMPORTANCE_CHANNELS,
            importance_factor=IMPORTANCE_FACTOR,
            importance_multiplier=IMPORTANCE_MULTIPLIER,
            regression_limits=REGRESSION_LIMITS,
            regression_reference=REGRESSION_REFERENCE,
            final_units=FINAL_UNITS,
            final_activation=FINAL_ACTIVATION,
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

        yield 'megan_2_normal', model

        model = Megan(
            units=UNITS,
            importance_channels=IMPORTANCE_CHANNELS,
            importance_factor=0.0,
            final_units=FINAL_UNITS,
            final_activation=FINAL_ACTIVATION,
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

        yield 'megan_2_supervised', model

    @se.hook('additional_metrics')
    def calculate_fidelity_ast(e, model, test_indices, x_test, y_test):
        # This metric can only be computed for MEGAN models, so we make sure that this is the case here
        if isinstance(model, Megan):
            e.info('calculating fidelity*, which is specific to multi-channel MEGAN models ...')
            importance_channels = e.parameters['IMPORTANCE_CHANNELS']
            rep = e['rep']
            key = e['key']

            out_pred, ni_pred, ei_pred = [v.numpy() for v in model(x_test)]

            # First of all we need to calculate the masked output deviations for each of the channels
            # and each of the elements in the dataset.
            for i in range(importance_channels):
                channel_direction = e.parameters['CHANNEL_DIRECTIONS'][i]

                # To do that we construct the corresponding masks here. For each channel we construct the
                # mask such that only that channel will be removed from the final pooling operation.
                node_masks = []
                for c, index in enumerate(test_indices):
                    ni = ni_pred[c]
                    node_mask = np.ones_like(ni)
                    node_mask[:, i] = 0
                    node_masks.append(node_mask)

                node_masks_tensor = ragged_tensor_from_nested_numpy(node_masks)
                # Using these tensors we can query the model again to get the modified predictions outputs
                out_mod, _, _ = [v.numpy() for v in model(x_test, node_importances_mask=node_masks_tensor)]
                for c, index in enumerate(test_indices):
                    value = channel_direction * (out_pred[c] - out_mod[c])
                    e[f'fidelity_ast_contribution/{rep}/{key}/{index}/{i}'] = value

            # Now we merge all the individual channel contributions into the final fidelity value
            for c, index in enumerate(test_indices):
                e[f'fidelity_ast/{rep}/{key}/{index}'] = np.sum([
                    e[f'fidelity_ast_contribution/{rep}/{key}/{index}/{i}']
                    for i in range(importance_channels)
                ])

            fidelities = [e[f'fidelity_ast/{rep}/{key}/{index}'] for index in test_indices]
            e.info(f'fidelity_ast results:\n'
                   f' * median: {np.median(fidelities):.3f}\n'
                   f' * mean: {np.mean(fidelities):.3f}\n'
                   f' * std: {np.std(fidelities):.3f}\n')

    # This hook adds the "fidelity_ast" metric results as an additional column to the final latex
    # evaluation table.
    @se.hook('additional_columns')
    def additional_columns(e):

        def fidelity_ast_callback(key, rep):
            return list(e[f'fidelity_ast/{rep}/{key}'].values())

        return [r'$\text{fidelity}^{*}$'], [fidelity_ast_callback]
