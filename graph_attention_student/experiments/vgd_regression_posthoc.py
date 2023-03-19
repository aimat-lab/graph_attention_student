"""

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
from graph_attention_student.models import GcnGradientModel
from graph_attention_student.models import grad_importances
from graph_attention_student.models import gnnx_importances
from graph_attention_student.training import LogProgressCallback
from graph_attention_student.training import mae
from graph_attention_student.util import array_normalize
from graph_attention_student.util import binary_threshold
from graph_attention_student.util import render_latex
from graph_attention_student.util import latex_table
from graph_attention_student.util import latex_table_element_mean

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
PATH = pathlib.Path(__file__).parent.absolute()

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
UNITS: t.List[int] = [12, 12, 12]
FINAL_UNITS: t.List[int] = [6, 1]
FINAL_ACTIVATION: str = 'linear'

# == EVALUATION PARAMETERS ==
# :param NUM_RANDOM_FIDELITY: This integer number determines the number of times which the "random fidelity"
#       is going to be sampled for each element of the dataset.
NUM_RANDOM_FIDELITY: int = 10
GNNX_EPOCHS: int = 200
GNNX_SPARSITY_FACTOR: float = 1.0

# == EXPERIMENT PARAMETERS ==
BASE_PATH = os.path.join(PATH, 'results')
NAMESPACE = os.path.basename(__file__).strip('.py')
DEBUG = True
TESTING = False

# If the TESTING flag is set that means that an execution of the experiment is supposed to be only a
# test of functionality. In that case we want to modify several parameters such that the length of the
# experiment is drastically reduced.
if TESTING:
    RATIO_DATASET = 0.2
    REPETITIONS = 1
    EPOCHS = 10
    NUM_RANDOM_MASKS = 1
    GNNX_EPOCHS = 10

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

        g['node_importances'] = g[NODE_IMPORTANCES_KEY]
        g['edge_importances'] = g[EDGE_IMPORTANCES_KEY]

        dataset[index] = g

    e.info(f'loaded dataset with {len(dataset)} elements')


    @e.hook('explanation_methods', default=True)
    def explanation_methods(_e,
                            model: ks.models.Model,
                            test_indices: t.List[int],
                            x_test: tuple,
                            y_test: tuple):

        # ~ Simple Gradient Explanations
        out_pred, node_gradient_info, edge_gradient_info = model(
            x_test,
            return_gradients=True,
            batch_size=len(test_indices),
        )
        ni_grad = grad_importances(
            gradient_info=node_gradient_info,
            use_absolute=True,
            keepdims=False,
        )
        ei_grad = grad_importances(
            gradient_info=edge_gradient_info,
            use_absolute=True,
            keepdims=False,
        )
        ni_grad = ni_grad.numpy()
        ei_grad = ei_grad.numpy()

        yield 'grad', ni_grad, ei_grad

        # ~ GNNExplainer
        ni_gnnx, ei_gnnx = gnnx_importances(
            model=lambda x: model(x, create_gradients=False, return_gradients=False),
            x=x_test,
            y=out_pred,
            epochs=_e.parameters['GNNX_EPOCHS'],
            node_sparsity_factor=_e.parameters['GNNX_SPARSITY_FACTOR'],
            edge_sparsity_factor=_e.parameters['GNNX_SPARSITY_FACTOR'],
            logger=_e.logger,
            log_step=10
        )
        ni_gnnx = ni_gnnx.numpy()
        ei_gnnx = ei_gnnx.numpy()

        yield 'gnnx', ni_gnnx, ei_gnnx

    e.info(f'starting explanations')
    for rep in range(REPETITIONS):
        e.info(f'REPETITION ({rep+1}/{REPETITIONS})')

        with tf.device(DEVICE):
            e.info('creating repetition archive folder...')
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

            e.info('creating the model...')
            model: ks.Model = GcnGradientModel(
                batch_size=BATCH_SIZE,
                units=UNITS,
                final_units=FINAL_UNITS,
                final_activation=FINAL_ACTIVATION,
                layer_cb=lambda units: AttentionHeadGATV2(
                    units=units,
                    use_edge_features=True,
                ),
                pooling_method='sum',
            )
            model.compile(
                loss=[ks.losses.MeanSquaredError()],
                loss_weights=[1],
                metrics=[ks.metrics.MeanSquaredError()],
                optimizer=OPTIMIZER_CB(),
                run_eagerly=False,
            )

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
            out_pred, node_gradient_info, edge_gradient_info = model(x_test, return_gradients=True)
            out_pred = out_pred.numpy()
            out_true, ni_true, ei_true = [v.numpy() if not isinstance(v, np.ndarray) else v
                                          for v in y_test]

            mse = mean_squared_error(out_true, out_pred)
            mae = mean_absolute_error(out_true, out_pred)
            r2 = r2_score(out_true, out_pred)
            e.info(f'prediction metrics:\n'
                   f' * mse: {mse:.3f}\n'
                   f' * mae: {mae:.3f}\n'
                   f' * r2: {r2:.3f}\n')

            e[f'mse/{rep}'] = mse
            e[f'mae/{rep}'] = mae
            e[f'r2/{rep}'] = r2

            for c, index in enumerate(test_indices):
                e[f'out/true/{rep}/{index}'] = out_true[c]
                e[f'out/pred/{rep}/{index}'] = out_pred[c]

            e['keys'] = []
            for key, ni_pred, ei_pred in e.apply_hook('explanation_methods',
                                                      model=model,
                                                      test_indices=test_indices,
                                                      x_test=x_test,
                                                      y_test=y_test):
                e.info(f'explanation method "{key}"...')
                e['keys'].append(key)

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
                node_mask_tensor = ragged_tensor_from_nested_numpy(node_masks)
                edge_mask_tensor = ragged_tensor_from_nested_numpy(edge_masks)
                out_masked = model((
                    x_test[0] * node_mask_tensor,
                    x_test[1] * edge_mask_tensor,
                    x_test[2]
                ), create_gradients=False, return_gradients=False)
                for c, index in enumerate(test_indices):
                    # Here we calculate the fidelity as the absolute difference in the predicted value.
                    # We multiply this with the fidelity of that element to make it a bit more comparable for
                    # the following reason: An explanation which consists of more explanations (and thus a
                    # bigger mask for the computation of the sparsity) will likely result in a bigger
                    # deviation naturally because more input elements are being perturbed.
                    fidelity = abs(out_masked[c] - out_pred[c])  # * e[f'node_sparsity/{key}/{rep}/{index}']
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
                    edge_mask_tensor = ragged_tensor_from_nested_numpy(edge_masks)

                    out_masked = model((
                        x_test[0] * node_mask_tensor,
                        x_test[1] * edge_mask_tensor,
                        x_test[2]
                    ), create_gradients=False, return_gradients=False)
                    for c, index in enumerate(test_indices):
                        fidelity = abs(out_masked[c] - out_pred[c])  # * e[f'node_sparsity/{key}/{rep}/{index}']
                        e[f'random_fidelity/{key}/{rep}/{index}/{j}'] = fidelity

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

    # ~ Latex table with the results over all repetitions
    e.info('rendering latex table with the results...')
    column_names = [
        r'Model Key',
        r'$\text{MSE} \downarrow$',
        r'$r^{2} \uparrow$',
        r'$\text{Node AUC} \uparrow$',
        r'$\text{Edge AUC} \uparrow$',
        r'$\text{Node Sparsity} \downarrow$',
        r'$\text{Edge Sparsity} \downarrow$',
        r'$\text{Fidelity} \uparrow$',
    ]
    rows = []
    for key in e['keys']:
        row = []
        row.append(key)
        row.append([e[f'mse/{rep}']
                    for rep in range(REPETITIONS)])
        row.append([e[f'r2/{rep}']
                    for rep in range(REPETITIONS)])
        row.append([e[f'node_auc/{key}/{rep}']
                    for rep in range(REPETITIONS)])
        row.append([e[f'edge_auc/{key}/{rep}']
                    for rep in range(REPETITIONS)])
        row.append([np.mean(list(e[f'node_sparsity/{key}/{rep}'].values()))
                    for rep in range(REPETITIONS)])
        row.append([np.mean(list(e[f'edge_sparsity/{key}/{rep}'].values()))
                    for rep in range(REPETITIONS)])
        row.append([np.mean(list(e[f'diff_fidelity/{key}/{rep}'].values()))
                    for rep in range(REPETITIONS)])

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
