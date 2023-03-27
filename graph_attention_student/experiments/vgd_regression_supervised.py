"""
Base experiment for training and comparing of multiple explanation supervised models specifically on graph
regression datasets.

This experiment will run multiple overall repetitions of model training processes to create a statistical
sample of the results. In each repetition multiple models are trained in an explanation supervised fashion
and then evaluated with respect to their prediction performance and their explanation accuracy towards the
ground truth explanations as well as the fidelity of their explanations.

**CHANGELOG**

0.1.0 - 18.03.2023 - Initial version
"""
import os
import pathlib
import random
import warnings
import typing as t

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as ks
from pycomex.util import Skippable
from pycomex.experiment import Experiment
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from kgcnn.data.utils import ragged_tensor_from_nested_numpy
from kgcnn.layers.conv.gat_conv import AttentionHeadGATV2
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.visualization.importances import create_importances_pdf
from graph_attention_student.data import process_graph_dataset
from graph_attention_student.models import Megan
from graph_attention_student.models import grad_importances
from graph_attention_student.models import GnesGradientModel
from graph_attention_student.training import LogProgressCallback
from graph_attention_student.training import NoLoss
from graph_attention_student.training import ExplanationLoss
from graph_attention_student.training import mae as mae_loss
from graph_attention_student.util import array_normalize
from graph_attention_student.util import binary_threshold
from graph_attention_student.util import render_latex
from graph_attention_student.util import latex_table
from graph_attention_student.util import latex_table_element_mean

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
PATH = pathlib.Path(__file__).parent.absolute()
SHORT_DESCRIPTION = (
    'Base experiment for training and comparing of multiple explanation supervised models specifically '
    'on graph regression datasets.'
)

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('~/.visual_graph_datasets/datasets/rb_dual_motifs')
RATIO_DATASET: t.Optional[float] = None
RATIO_TRAIN: float = 0.8
NODE_IMPORTANCES_KEY = 'node_importances_1'
EDGE_IMPORTANCES_KEY = 'edge_importances_1'
NUM_EXAMPLE: int = 100

# == TRAINING PARAMETERS ==
REPETITIONS = 25
DEVICE = 'cpu:0'
BATCH_SIZE = 32
OPTIMIZER_CB = lambda: ks.optimizers.Adam(learning_rate=0.01)
EPOCHS = 25

# == MODEL PARAMETERS ==
UNITS: t.List[int] = [32, 32, 32]
FINAL_UNITS: t.List[int] = [16, 1]
FINAL_ACTIVATION: str = 'linear'
SPARSITY_FACTOR: float = 1.0

MEGAN_UNITS: t.List[int] = [12, 12, 12]
MEGAN_FINAL_UNITS: t.List[int] = [6, 1]
MEGAN_IMPORTANCE_CHANNELS: int = 1

# == EVALUATION PARAMETERS ==
BATCH_SIZE_EVAL = 128
# :param NUM_RANDOM_FIDELITY: This integer number determines the number of times which the "random fidelity"
#       is going to be sampled for each element of the dataset.
NUM_RANDOM_FIDELITY: int = 10

# == EXPERIMENT PARAMETERS ==
BASE_PATH = os.path.join(PATH, 'results')
NAMESPACE = 'vgd_regression_supervised'
DEBUG = True
TESTING = False

# If the TESTING flag is set that means that an execution of the experiment is supposed to be only a
# test of functionality. In that case we want to modify several parameters such that the length of the
# experiment is drastically reduced.
if TESTING:
    RATIO_DATASET = 0.2
    REPETITIONS = 1
    EPOCHS = 10
    NUM_RANDOM_FIDELITY = 1
    NUM_EXAMPLE = 20

with Skippable(), (e := Experiment(BASE_PATH, NAMESPACE, globals())):
    vgd_name = os.path.basename(VISUAL_GRAPH_DATASET_PATH)
    e.info(f'using visual graph dataset "{vgd_name}"')

    # ~ loading the dataset
    e.info('loading visual graph dataset from disk')
    metadata_map, index_data_map = load_visual_graph_dataset(
        VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=1000,
        metadata_contains_index=True,
    )
    dataset_indices_all = list(sorted(index_data_map.keys()))
    dataset_length = len(dataset_indices_all)

    dataset: t.List[dict] = [None for _ in dataset_indices_all]
    for index in dataset_indices_all:
        g = index_data_map[index]['metadata']['graph']

        g['node_importances'] = np.array(g[NODE_IMPORTANCES_KEY]).astype(float)
        g['edge_importances'] = np.array(g[EDGE_IMPORTANCES_KEY]).astype(float)

        # We explicitly cast the attribute vectors as a float type as that is important for certain models
        # to work properly down the line
        g['node_attributes'] = np.array(g['node_attributes']).astype(float)
        g['edge_attributes'] = np.array(g['edge_attributes']).astype(float)

        dataset[index] = g

    e.info(f'loaded dataset with {len(dataset)} elements')

    @e.hook('model_generator')
    def model_generator(_e):

        # ~ fixed GNES version
        # This versions uses the absolute activation, which works better for regression problems
        model = GnesGradientModel(
            units=_e.parameters['UNITS'],
            batch_size=_e.parameters['BATCH_SIZE'],
            final_units=_e.parameters['FINAL_UNITS'],
            final_activation=_e.parameters['FINAL_ACTIVATION'],
            layer_cb=lambda units: AttentionHeadGATV2(
                units=units,
                use_edge_features=True,
            ),
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
            units=_e.parameters['UNITS'],
            batch_size=_e.parameters['BATCH_SIZE'],
            final_units=_e.parameters['FINAL_UNITS'],
            final_activation=_e.parameters['FINAL_ACTIVATION'],
            layer_cb=lambda units: AttentionHeadGATV2(
                units=units,
                use_edge_features=True,
            ),
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
            units=MEGAN_UNITS,
            importance_channels=MEGAN_IMPORTANCE_CHANNELS,
            importance_factor=0.0,
            sparsity_factor=SPARSITY_FACTOR,
            concat_heads=False,
            final_units=MEGAN_FINAL_UNITS,
            final_activation=FINAL_ACTIVATION,
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
        yield 'megan_normal', model

        # ~ MEGAN supervised
        # This MEGAN version does not receive explanation supervision
        model = Megan(
            units=MEGAN_UNITS,
            importance_channels=MEGAN_IMPORTANCE_CHANNELS,
            importance_factor=0.0,
            sparsity_factor=SPARSITY_FACTOR,
            concat_heads=False,
            final_units=MEGAN_FINAL_UNITS,
            final_activation=FINAL_ACTIVATION,
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
        yield 'megan_supervised', model

    # -- MAIN LOOP --
    e.info(f'starting repetition loop...')
    for rep in range(REPETITIONS):
        e.info(f'REPETITION ({rep+1}/{REPETITIONS})')

        with tf.device(DEVICE):
            e.info('creating rep archive folder...')
            rep_path = os.path.join(e.path, f'{rep:02d}')
            os.mkdir(rep_path)

            e.info('creating random dataset split...')
            dataset_indices = dataset_indices_all.copy()
            if RATIO_DATASET:
                num_elements = int(RATIO_DATASET * dataset_length)
                dataset_indices = dataset_indices[:num_elements]

            dataset_indices_set = set(dataset_indices)
            num_train = int(len(dataset_indices) * RATIO_TRAIN)
            train_indices = random.sample(dataset_indices, k=num_train)
            train_indices_set = set(train_indices)
            test_indices_set = dataset_indices_set.difference(train_indices_set)
            test_indices = list(test_indices_set)

            num_example = min(len(test_indices), NUM_EXAMPLE)
            example_indices = random.sample(test_indices, k=num_example)

            e[f'train_indices/{rep}'] = train_indices
            e[f'test_indices/{rep}'] = test_indices
            e[f'example_indices/{rep}'] = example_indices
            e.info(f'created split with '
                   f'{len(train_indices)} train elements, '
                   f'{len(test_indices)} test elements, '
                   f'{len(example_indices)} example elements for visualization')

            e.info('processing the dataset into tensors...')
            x_train, y_train, x_test, y_test = process_graph_dataset(
                dataset,
                test_indices=test_indices,
                train_indices=train_indices,
                use_importances=True,
                use_graph_attributes=False,
            )
            e['keys'] = []
            for key, model in e.apply_hook('model_generator'):
                e.info(f'creating the model "{key}"...')
                # Here we update the current repetition index and model key into the experiment storage
                # so that this can be accessed by sub experiment hook implementations without needing
                # to explicitly pass that information every time.
                e['rep'] = rep
                e['key'] = key
                e['keys'].append(key)

                e.info('starting model training...')
                hist = model.fit(
                    x_train,
                    y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=[LogProgressCallback(
                        logger=e.logger,
                        epoch_step=1,
                        identifier=f'loss'
                    )],
                    verbose=0,
                )
                history = hist.history
                e[f'history/{rep}'] = history
                e.info(f'finished training for model with {model.count_params()} parameters')

                e.info('evaluating on test set...')

                # ~ The overall prediction metrics
                out_pred, ni_pred, ei_pred = [v.numpy()
                                              for v in model(x_test, batch_size=len(test_indices))]
                out_true, ni_true, ei_true = [v.numpy() if not isinstance(v, np.ndarray) else v
                                              for v in y_test]

                mse = mean_squared_error(out_true, out_pred)
                mae = mean_absolute_error(out_true, out_pred)
                r2 = r2_score(out_true, out_pred)
                e.info(f'prediction metrics:\n'
                       f' * mse: {mse:.3f}\n'
                       f' * mae: {mae:.3f}\n'
                       f' * r2: {r2:.3f}\n')

                e[f'mse/{key}/{rep}'] = mse
                e[f'mae/{key}/{rep}'] = mae
                e[f'r2/{key}/{rep}'] = r2

                for c, index in enumerate(test_indices):
                    e[f'out/true/{rep}/{index}'] = out_true[c]
                    e[f'out/pred/{key}/{rep}/{index}'] = out_pred[c]

                node_masks = []
                edge_masks = []
                for c, index in enumerate(test_indices):
                    ni = array_normalize(ni_pred[c])
                    ei = array_normalize(ei_pred[c])
                    e[f'ni/{key}/{rep}/{index}'] = ni
                    e[f'ei/{key}/{rep}/{index}'] = ei

                    # Up until this point, every importance value is still a float, but we need to the binary
                    # explanations for two reasons: To construct the masks for the fidelity computation and
                    # for the computation of the sparsity
                    ni_binary = binary_threshold(ni, threshold=0.5)
                    ei_binary = binary_threshold(ei, threshold=0.5)
                    node_masks.append(1.0 - ni_binary.astype(float))
                    edge_masks.append(1.0 - ei_binary.astype(float))

                    node_sparsity = np.mean(ni_binary)
                    edge_sparsity = np.mean(ei_binary)
                    e[f'node_sparsity/{key}/{rep}/{index}'] = node_sparsity
                    e[f'edge_sparsity/{key}/{rep}/{index}'] = edge_sparsity

                e.info('calculating fidelity...')

                # Why do we sum here as well?
                # Technically the last dimension of the masks will be "1" anyway so the sum reduce has
                # absolutely no effect. We do this to eventually support multi channel explanations, which
                # have a last dimension != 1, as well, so that they don't cause an error at this point.
                node_mask_tensor = ragged_tensor_from_nested_numpy(node_masks)
                node_mask_tensor = tf.reduce_sum(node_mask_tensor, axis=-1, keepdims=True)
                edge_mask_tensor = ragged_tensor_from_nested_numpy(edge_masks)
                edge_mask_tensor = tf.reduce_sum(edge_mask_tensor, axis=-1, keepdims=True)

                out_masked, _, _ = model((
                    x_test[0] * node_mask_tensor,
                    x_test[1] * edge_mask_tensor,
                    x_test[2]
                ), create_gradients=False)
                for c, index in enumerate(test_indices):
                    # Here we calculate the fidelity as the absolute difference in the predicted value.
                    fidelity = abs(out_masked[c] - out_pred[c])
                    e[f'fidelity/{key}/{rep}/{index}'] = fidelity

                e.info('calculating random fidelity...')
                for j in range(NUM_RANDOM_FIDELITY):
                    # first of all we need to create the random masks, which we can very easily do by simply
                    # using the "correct" explanation masks we already have and simply randomly permutating
                    # them, because one important property for the random masks is that they have to have
                    # the same sparsity!
                    for node_mask, edge_mask in zip(node_masks, edge_masks):
                        np.random.shuffle(node_mask)
                        np.random.shuffle(edge_mask)

                    node_mask_tensor = ragged_tensor_from_nested_numpy(node_masks)
                    node_mask_tensor = tf.reduce_sum(node_mask_tensor, axis=-1, keepdims=True)
                    edge_mask_tensor = ragged_tensor_from_nested_numpy(edge_masks)
                    edge_mask_tensor = tf.reduce_sum(edge_mask_tensor, axis=-1, keepdims=True)

                    out_masked, _, _ = model((
                        x_test[0] * node_mask_tensor,
                        x_test[1] * edge_mask_tensor,
                        x_test[2]
                    ), create_gradients=False)
                    for c, index in enumerate(test_indices):
                        fidelity = abs(out_masked[c] - out_pred[c])  # * e[f'node_sparsity/{key}/{rep}/{index}']
                        e[f'random_fidelity/{key}/{rep}/{index}/{j}'] = fidelity

                # :hook additional_metrics:
                #       This is an action hook which can be used to inject the calculation of additional
                #       metrics, since it does not return anything to the main runtime, the results will
                #       have to be directly stored to the internal experiment storage.
                e.apply_hook(
                    'additional_metrics',
                    model=model,
                    test_indices=test_indices,
                    x_test=x_test,
                    y_test=y_test
                )

                # ~ calculating the explanation specific metrics
                node_auc = roc_auc_score(
                    np.concatenate([v.flatten() for v in ni_true]),
                    np.concatenate([v.flatten() for v in ni_pred]),
                )
                edge_auc = roc_auc_score(
                    np.concatenate([v.flatten() for v in ei_true]),
                    np.concatenate([v.flatten() for v in ei_pred]),
                )
                e[f'node_auc/{key}/{rep}'] = node_auc
                e[f'edge_auc/{key}/{rep}'] = edge_auc

                node_sparsities = [e[f'node_sparsity/{key}/{rep}/{index}'] for index in test_indices]
                edge_sparsities = [e[f'edge_sparsity/{key}/{rep}/{index}'] for index in test_indices]

                fidelities = np.array([e[f'fidelity/{key}/{rep}/{index}'] for index in test_indices])
                fidelity_mean = np.mean(fidelities)
                fidelity_std = np.std(fidelities)
                random_fidelities = np.array([np.mean([e[f'random_fidelity/{key}/{rep}/{index}/{j}']
                                                       for j in range(NUM_RANDOM_FIDELITY)])
                                              for index in test_indices])
                diff_fidelities = fidelities - random_fidelities
                for c, index in enumerate(test_indices):
                    e[f'rand_fidelity/{key}/{rep}/{index}'] = random_fidelities[c]
                    e[f'diff_fidelity/{key}/{rep}/{index}'] = diff_fidelities[c]

                e.info(f'explanation metrics:\n'
                       f' * node auc: {node_auc:.3f}\n'
                       f' * edge auc: {edge_auc:.3f}\n'
                       f' * mean node sparsity: {np.mean(node_sparsities):.3f}\n'
                       f' * mean edge sparsity: {np.mean(edge_sparsities):.3f}\n'
                       f' * mean fidelity: {fidelity_mean:.3f}\n'
                       f' * std fidelity: {fidelity_std:.3f}\n'
                       f' * mean random fidelity: {np.mean(random_fidelities):.3f}\n'
                       f' * mean diff fidelity: {np.mean(diff_fidelities):.3f}\n'
                       )

            e.info('visualizing examples...')
            graph_list = [index_data_map[i]['metadata']['graph'] for i in example_indices]
            image_path_list = [index_data_map[i]['image_path'] for i in example_indices]
            node_positions_list = [g['node_positions'] for g in graph_list]
            output_path = os.path.join(rep_path, 'examples.pdf')
            create_importances_pdf(
                graph_list=graph_list,
                image_path_list=image_path_list,
                node_positions_list=node_positions_list,
                importances_map={
                    'gt': (
                        [g['node_importances'] for g in graph_list],
                        [g['edge_importances'] for g in graph_list],
                    ),
                    **{
                        key: (
                            [e[f'ni/{key}/{rep}/{index}'] for index in example_indices],
                            [e[f'ei/{key}/{rep}/{index}'] for index in example_indices]
                        )
                        for key in e['keys']
                    }
                },
                output_path=output_path,
                logger=e.logger,
                log_step=100,
            )


with Skippable(), e.analysis:

    e.info('starting analysis of experiment results...')

    e.info('cleaning non-convergent repetitions...')
    REPETITIONS_CLEAN = []
    for rep in range(REPETITIONS):
        use_repetition = True
        for key in e['keys']:
            value = e[f'mse/{key}/{rep}']
            if value > 2.0:
                use_repetition = False

        if use_repetition:
            REPETITIONS_CLEAN.append(rep)

    e.info(f'using {len(REPETITIONS_CLEAN)} repetitions...')

    # ~ Latex table with the results over all repetitions
    e.info('rendering latex table with the results...')

    # :hook additional_columns:
    #       This hook can be used to add additional columns to the final latex evaluation table from
    #       within the child experiments. It has to return a tuple of two lists of same size.
    #       the first is a list of string names for the columns and the second list is a list of
    #       callback functions, which if given the repetition index and the key, return all the values
    #       for the corresponding metric to be used as a base for the table columns contents.
    additional_column_names, additional_callbacks = e.apply_hook('additional_columns', default=([], []))

    column_names = [
        r'Model Key',
        r'$\text{MSE} \downarrow$',
        r'$r^{2} \uparrow$',
        r'$\text{Node AUC} \uparrow$',
        r'$\text{Edge AUC} \uparrow$',
        r'$\text{Node Sparsity} \downarrow$',
        r'$\text{Edge Sparsity} \downarrow$',
        r'$\text{Fidelity} \uparrow$',
    ] + additional_column_names
    rows = []
    for key in e['keys']:
        row = []
        row.append(key.replace('_', ' '))
        row.append([e[f'mse/{key}/{rep}']
                    for rep in REPETITIONS_CLEAN])
        row.append([e[f'r2/{key}/{rep}']
                    for rep in REPETITIONS_CLEAN])
        row.append([e[f'node_auc/{key}/{rep}']
                    for rep in REPETITIONS_CLEAN])
        row.append([e[f'edge_auc/{key}/{rep}']
                    for rep in REPETITIONS_CLEAN])
        row.append([np.mean(list(e[f'node_sparsity/{key}/{rep}'].values()))
                    for rep in REPETITIONS_CLEAN])
        row.append([np.mean(list(e[f'edge_sparsity/{key}/{rep}'].values()))
                    for rep in REPETITIONS_CLEAN])
        row.append([np.mean(list(e[f'diff_fidelity/{key}/{rep}'].values()))
                    for rep in REPETITIONS_CLEAN])

        # Each one of the additional callbacks is supposed to return a list of all the values corresponding
        # to the metrics for exactly that key and that repetition
        for cb in additional_callbacks:
            row.append([np.mean(cb(key, rep)) for rep in REPETITIONS_CLEAN])

        rows.append(row)

    content, table = latex_table(
        column_names=column_names,
        rows=rows,
        list_element_cb=latex_table_element_mean,
    )
    e.commit_raw('table.tex', table)
    output_path = os.path.join(e.path, 'table.pdf')
    render_latex({'content': table}, output_path=output_path)
    e.info('rendered latex table with the results')
