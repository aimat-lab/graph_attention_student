"""
Inherits ""

**CHANGLELOG**

0.1.0 - xx.xx.2023 - Initial version
"""
import os
import pathlib
import typing as t

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from pycomex.util import Skippable
from pycomex.experiment import SubExperiment
from kgcnn.data.utils import ragged_tensor_from_nested_numpy
from kgcnn.literature.DMPNN import make_model

from graph_attention_student.training import mse
from graph_attention_student.training import LogProgressCallback

# == MODEL PARAMETERS ==
MODEL_NAME = 'DMPNN'

# == TRAINING PARAMETERS ==
BATCH_SIZE = 64
OPTIMIZER_CB = lambda: ks.optimizers.Nadam(learning_rate=0.001)

# == EXPERIMENT PARAMETERS ==
PATH = pathlib.Path(__file__).parent.absolute()
EXPERIMENT_PATH = os.path.join(PATH, 'vgd_single_gnnx_aqsoldb.py')
BASE_PATH = PATH
NAMESPACE = 'results/vgd_single_gnnx_aqsoldb_dmpnn'
with Skippable(), (se := SubExperiment(EXPERIMENT_PATH, BASE_PATH, NAMESPACE, globals())):

    @se.hook('process_dataset')
    def process_dataset(e, dataset, train_indices, test_indices):
        node_attributes_train = [dataset[i]['node_attributes'] for i in train_indices]
        node_attributes_test = [dataset[i]['node_attributes'] for i in test_indices]
        e['num_node_attributes'] = dataset[0]['node_attributes'].shape[1]

        edge_attributes_train = [dataset[i]['edge_attributes'] for i in train_indices]
        edge_attributes_test = [dataset[i]['edge_attributes'] for i in test_indices]
        e['num_edge_attributes'] = dataset[0]['edge_attributes'].shape[1]

        edge_indices_train = [dataset[i]['edge_indices'] for i in train_indices]
        edge_indices_test = [dataset[i]['edge_indices'] for i in test_indices]

        edge_indices_reverse_train = [np.expand_dims(dataset[i]['edge_indices'][:, 1], -1) for i in train_indices]
        edge_indices_reverse_test = [np.expand_dims(dataset[i]['edge_indices'][:, 1], -1) for i in test_indices]

        graph_labels_train = [dataset[i]['graph_labels'] for i in train_indices]
        graph_labels_test = [dataset[i]['graph_labels'] for i in test_indices]

        node_importances_train = [dataset[i]['node_importances'] for i in train_indices]
        node_importances_test = [dataset[i]['node_importances'] for i in test_indices]

        edge_importances_train = [dataset[i]['edge_importances'] for i in train_indices]
        edge_importances_test = [dataset[i]['edge_importances'] for i in test_indices]

        x_train = (
            ragged_tensor_from_nested_numpy(node_attributes_train),
            ragged_tensor_from_nested_numpy(edge_attributes_train),
            ragged_tensor_from_nested_numpy(edge_indices_train),
            ragged_tensor_from_nested_numpy(edge_indices_reverse_train)
        )

        x_test = (
            ragged_tensor_from_nested_numpy(node_attributes_test),
            ragged_tensor_from_nested_numpy(edge_attributes_test),
            ragged_tensor_from_nested_numpy(edge_indices_test),
            ragged_tensor_from_nested_numpy(edge_indices_reverse_test)
        )

        y_train = (
            np.array(graph_labels_train),
            ragged_tensor_from_nested_numpy(node_importances_train),
            ragged_tensor_from_nested_numpy(edge_importances_train),
        )

        y_test = (
            np.array(graph_labels_test),
            ragged_tensor_from_nested_numpy(node_importances_test),
            ragged_tensor_from_nested_numpy(edge_importances_test),
        )

        return x_train, y_train, x_test, y_test

    @se.hook('create_model')
    def create_model(e):
        model = make_model(
            inputs=[
                {'shape': (None, e['num_node_attributes']), 'name': 'node_attributes', 'dtype': 'float32', 'ragged': True},
                {'shape': (None, e['num_edge_attributes']), 'name': 'edge_attributes', 'dtype': 'float32', 'ragged': True},
                {'shape': (None, 2), 'name': 'edge_indices', 'dtype': 'int64', 'ragged': True},
                {'shape': (None, 1), 'name': 'edge_indices_reverse', 'dtype': 'int64', 'ragged': True}
            ],
            input_embedding={},
            pooling_args={'pooling_method': 'sum'},
            use_graph_state=False,
            edge_initialize={'units': 64, 'use_bias': True, 'activation': 'relu'},
            edge_dense={'units': 64, 'use_bias': True, 'activation': 'linear'},
            edge_activation={'activation': 'relu'},
            node_dense={'units': 64, 'use_bias': True, 'activation': 'relu'},
            depth=5,
            dropout={'rate': 0.1},
            output_mlp={"use_bias": [True, True, False], "units": [64, 32, 1],
                        "activation": ["relu", "relu", "linear"]}
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
