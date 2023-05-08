import os
import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as ks
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.visualization.importances import create_importances_pdf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from kgcnn.data.utils import ragged_tensor_from_nested_numpy

from graph_attention_student.models.megan import Megan
from graph_attention_student.data import process_index_data_map
from graph_attention_student.data import process_graph_dataset
from graph_attention_student.training import LogProgressCallback
from graph_attention_student.training import mse, NoLoss
from graph_attention_student.visualization import plot_regression_fit
from graph_attention_student.util import array_normalize

VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('~/.visual_graph_datasets/datasets/organic_solvents')
NUM_TARGETS = 4
TRAIN_RATIO = 0.8
NUM_EXAMPLES = 100

UNITS = [32, 32, 32]
DROPOUT_RATE = 0.15
FINAL_UNITS = [32, 16, NUM_TARGETS]
FINAL_DROPOUT_RATE = 0.05
IMPORTANCE_CHANNELS = 4
FINAL_ACTIVATION = 'linear'
SPARSITY_FACTOR = 0.5
GINI_FACTOR = 3.0
EPOCHS = 100
BATCH_SIZE = 32

# == VISUALIZATION PARAMETERS ==
BASE_FIG_SIZE: int = 10

__DEBUG__ = True
__TESTING__ = False


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    e.log('starting experiment for a multi regression MEGAN model...')

    if e.__TESTING__:
        e.log('TESTING MODE!')
        e.EPOCHS = 1
        e.NUM_EXAMPLES = 10

    e.log('loading the dataset...')
    e.log(f' * path: {e.VISUAL_GRAPH_DATASET_PATH}')
    metadata_map, index_data_map = load_visual_graph_dataset(
        path=e.VISUAL_GRAPH_DATASET_PATH,
        metadata_contains_index=True,
        logger=e.logger,
        log_step=1000,
    )

    num_targets = len(index_data_map[0]["metadata"]["target"])
    target_count_map = defaultdict(int)
    for index, data in index_data_map.items():
        for target_index in range(num_targets):
            if data['metadata']['target'][target_index] is not None:
                target_count_map[target_index] += 1

    e.log(f'number of target values: {num_targets}')
    for target_index, count in target_count_map.items():
        e.log(f' * {target_index}: {count}')

    dataset_indices, graph_list = process_index_data_map(
        index_data_map,
        insert_empty_importances=True,
        importance_channels=e.IMPORTANCE_CHANNELS
    )
    k_train = int(len(dataset_indices) * e.TRAIN_RATIO)
    train_indices = random.sample(dataset_indices, k=k_train)
    test_indices = list(set(dataset_indices).difference(set(train_indices)))
    k_examples = min(NUM_EXAMPLES, len(test_indices))
    example_indices = random.sample(test_indices, k=k_examples)

    x_train, y_train, x_test, y_test = process_graph_dataset(
        graph_list,
        train_indices=train_indices,
        test_indices=test_indices,
        use_importances=True,
        use_graph_attributes=False,
    )

    # ~ Setting up the model

    @e.hook('create_model')
    def create_model(e):
        model = Megan(
            units=e.UNITS,
            dropout_rate=e.DROPOUT_RATE,
            final_units=e.FINAL_UNITS,
            final_dropout_rate=e.FINAL_DROPOUT_RATE,
            final_activation=e.FINAL_ACTIVATION,
            importance_channels=e.IMPORTANCE_CHANNELS,
            importance_factor=0,
            importance_multiplier=0,
            sparsity_factor=e.SPARSITY_FACTOR,
            gini_factor=e.GINI_FACTOR,
            concat_heads=False,
            use_graph_attributes=False,
        )
        model.compile(
            optimizer=ks.optimizers.Adam(learning_rate=0.001),
            loss=[mse, NoLoss(), NoLoss()],
            loss_weights=[1, 0, 0],
            metrics=[mse],
            run_eagerly=False,
        )

        return model

    e.log('creating model...')
    model = e.apply_hook('create_model')

    # ~ Starting the training
    hist = model.fit(
        x_train, y_train,
        epochs=e.EPOCHS,
        batch_size=e.BATCH_SIZE,
        validation_freq=1,
        validation_data=(x_test, y_test),
        callbacks=LogProgressCallback(
            identifier='val_output_1_mean_squared_error',
            logger=e.logger,
            epoch_step=5,
        ),
        verbose=0,
    )
    history = hist.history

    # ~ evaluating on test set
    out_true, ni_true, ei_true = [v.numpy() if not isinstance(v, np.ndarray) else v for v in y_test]
    out_pred, ni_pred, ei_pred = [v.numpy() if not isinstance(v, np.ndarray) else v for v in model(x_test)]
    for c, index in enumerate(test_indices):
        out = out_pred[c]
        ni = array_normalize(ni_pred[c])
        ei = array_normalize(ei_pred[c])

        e[f'out/pred/{index}'] = out
        e[f'ni/pred/{index}'] = ni
        e[f'ei/pred/{index}'] = ei

    fig_test, rows_test = plt.subplots(
        ncols=e.NUM_TARGETS,
        nrows=1,
        figsize=(e.NUM_TARGETS * e.BASE_FIG_SIZE, e.BASE_FIG_SIZE),
        squeeze=False,
    )
    for target_index in range(e.NUM_TARGETS):
        e.log(f'target index: {target_index}')
        # ~ test set results
        # First of all we calculate the actual metrics for each of the targets separately
        valid_indices_test = [i for i, v in enumerate(out_true) if not bool(np.isnan(v[target_index]))]
        e.log(f' * num test elements: {len(valid_indices_test)}')
        values_true = [out_true[i][target_index] for i in valid_indices_test]
        values_pred = [out_pred[i][target_index] for i in valid_indices_test]

        mse_value = mean_squared_error(values_true, values_pred)
        r2_value = r2_score(values_true, values_pred)
        e[f'mse/test/{target_index}'] = mse_value
        e[f'r2/test/{target_index}'] = r2_value
        e.log(f' * mse: {mse_value:.2f}')
        e.log(f' * r2: {r2_value:.2f}')

        ax_test = rows_test[0][target_index]
        plot_regression_fit(
            values_true=values_true,
            values_pred=values_pred,
            ax=ax_test
        )
        ax_test.set_title(f'Target: {target_index} \n'
                          f'MSE: {mse_value:.3f}\n'
                          f'R2: {r2_value:.3f}')

    fig_test.savefig(os.path.join(e.path, 'test_results.pdf'))
    plt.close(fig_test)

    # ~ Analyzing the explanations
    e.log('analyzing the explanation channels... beware that this might take some time')

    # ~ Leave one out
    e.log('starting leave-one-out analysis...')
    for channel_index in range(e.IMPORTANCE_CHANNELS):
        base_mask = [float(channel_index != j) for j in range(e.IMPORTANCE_CHANNELS)]
        mask_one_out = [
            [base_mask for _ in index_data_map[i]['metadata']['graph']['node_indices']]
            for i in test_indices
        ]
        out_mod, _, _ = model(
            x_test,
            node_importances_mask=ragged_tensor_from_nested_numpy(mask_one_out)
        )

        for target_index in range(e.NUM_TARGETS):
            for c, index in enumerate(test_indices):
                mod = out_mod[c][target_index] - e[f'out/pred/{index}'][target_index]
                e[f'out/mod/one_out/{index}/{target_index}/{channel_index}'] = mod

    fig_one_out, rows = plt.subplots(
        nrows=e.IMPORTANCE_CHANNELS,
        ncols=e.NUM_TARGETS,
        figsize=(e.BASE_FIG_SIZE * e.NUM_TARGETS, e.BASE_FIG_SIZE * e.IMPORTANCE_CHANNELS),
        squeeze=False,
    )
    for channel_index in range(e.IMPORTANCE_CHANNELS):
        max_value = 0
        min_value = 0
        for target_index in range(e.NUM_TARGETS):
            ax: plt.Axes = rows[channel_index][target_index]
            ax.set_title(f'Target: {target_index} - Channel: {channel_index}')
            values = [e[f'out/mod/one_out/{index}/{target_index}/{channel_index}'] for index in test_indices]
            max_value = max(max_value, *values)
            min_value = min(min_value, *values)
            ax.hist(
                values,
                bins=20,
                color='lightgray',
            )

        for target_index in range(e.NUM_TARGETS):
            ax: plt.Axes = rows[channel_index][target_index]
            ax.set_xlim([min_value, max_value])

    fig_one_out.savefig(os.path.join(e.path, 'leave_one_out.pdf'))
    plt.close(fig_one_out)

    # ~ visualizing the explanations
    output_path = os.path.join(e.path, 'explanations.pdf')
    graph_list = [index_data_map[i]['metadata']['graph'] for i in example_indices]
    create_importances_pdf(
        graph_list=graph_list,
        image_path_list=[index_data_map[i]['image_path'] for i in example_indices],
        node_positions_list=[g['node_positions'] for g in graph_list],
        output_path=output_path,
        importances_map={
          'predicted': (
              [e[f'ni/pred/{i}'] for i in example_indices],
              [e[f'ei/pred/{i}'] for i in example_indices]
          )
        },
        logger=e.logger,
        log_step=20,
    )


experiment.run_if_main()
