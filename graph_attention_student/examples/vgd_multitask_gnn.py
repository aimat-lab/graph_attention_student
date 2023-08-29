"""
This example shows how to train a KGCNN model for multitask graph regression application based on an
existing "visual graph dataset".

This module loads a multitask regression visual graph dataset and trains a multi-layered GNN model to
predict all the targets simultaneously. This base module uses standard GCN layers and a global pooling
operation followed by a MLP to make the prediction.
The module calculates the most important regression performance metrics for each of the targets, such as
MSE and R2 and plots these results for each repetition of the training process.

This module implements various hooks, which means it can be used as the basis for various modifications,
such as changing the used model, by sub-experiment implementation.

**CHANGELOG**

0.1.0 - 17.01.2023 - Initial version

0.2.0 - 18.01.2023 - The experiment can now be repeated multiple independent times with different train
test splits of the dataset.
Analysis now renders a latex table containing the combined results of all the different targets
"""
# standard library
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import typing as t

# third party
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as ks
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.metrics import mean_absolute_error as mae_score
from pycomex.experiment import Experiment
from pycomex.util import Skippable
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from visual_graph_datasets.data import load_visual_graph_dataset
from kgcnn.layers.conv.gcn_conv import GCN
from kgcnn.layers.pooling import PoolingGlobalEdges
from kgcnn.layers.modules import DenseEmbedding, DropoutEmbedding, LazyConcatenate

# local
import graph_attention_student.typing as tc
from graph_attention_student.data import process_graph_dataset
from graph_attention_student.training import mse
from graph_attention_student.training import LogProgressCallback
from graph_attention_student.visualization import plot_regression_fit
from graph_attention_student.util import latex_table, latex_table_element_mean
from graph_attention_student.util import render_latex


# == DATASET PARAMETERS ==
# These parameters are used to specify the dataset to be used for the training as well as additional
# properties of the dataset such as the train test split for example.

# The name of the visual graph dataset to use for this experiment.
VISUAL_GRAPH_DATASET_PATH = os.path.expanduser('~/.visual_graph_datasets/datasets/organic_solvents')
# The ratio of how many elements of the dataset are supposed to be used for the training dataset.
# The rest of them will be used for the test set.
TRAIN_RATIO = 0.8
# The number of target values for each graph in the dataset.
NUM_TARGETS = 4
# Whether the dataset already contains importance (explanation) ground truth annotations.
# most of the time this will most likely not be the case
HAS_IMPORTANCES: bool = False
# IF the dataset includes global graph attributes and if they are supposed to be used in the training
# process, this flag has to be set to True.
USE_GRAPH_ATTRIBUTES: bool = True
# The ratio of the test set to be used as examples for the visualization of the explanations
EXAMPLES_RATIO: float = 0.2
# The string names of the target values in the order in which they appear in the dataset as well
# which will be used in the labels for the result visualizations
TARGET_NAMES: t.List[str] = [
    'water',
    'benzene',
    'acetone',
    'ethanol',
]


# == MODEL PARAMETERS ==

class GcnModel(ks.models.Model):

    def __init__(self,
                 units: t.List[int],
                 final_units: t.List[int],
                 pooling_method: str = 'sum',
                 activation: str = 'kgcnn>leaky_relu',
                 final_pooling: str = 'sum',
                 final_activation: str = 'linear',
                 dropout_rate: float = 0.0,
                 use_graph_attributes: bool = False):
        super(GcnModel, self).__init__()
        self.use_graph_attributes = use_graph_attributes

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
        if self.use_graph_attributes:
            node_input, edge_input, edge_index_input, graph_input = inputs
        else:
            node_input, edge_input, edge_index_input = inputs

        x = node_input
        for lay in self.conv_layers:
            x = lay([x, edge_input, edge_index_input])
            if training:
                x = self.lay_dropout(x)

        final = self.lay_pooling(x)
        if self.use_graph_attributes:
            final = self.lay_concat([final, graph_input])

        for lay in self.final_layers:
            final = lay(final)

        return final


MODEL_NAME = 'GCN'
UNITS = [64, 64, 64]
FINAL_UNITS = [30, 10]
DROPOUT_RATE = 0.2

# == TRAINING PARAMETERS ==
# These parameters are for the learning / training process of the model.

# The training can be repeated multiple times to get a statistical measure for the performance of the model.
# This determines how many repetitions are made
REPETITIONS: int = 1
# This string defines which device is used for the training process. On a normal setup for a normal PC the
# main two options would be 'cpu:0' and 'gpu:0' which uses the CPU and GPU respectively. Training on a
# GPU is usually more efficient, but one might encounter errors in some cases such as tensor operations
# which are not possible on GPU or insufficient video memory for large batches. In such a case one should
# switch to CPU instead, which will be slower.
DEVICE: str = 'gpu:0'
# This is the learning rate of the optimization process used by the Optimizer.
LEARNING_RATE: float = 0.01
# The batch size is the number of elements from the training dataset which will be concatenated into a
# single tensor, which will then be used for one forward & backward pass of the model, ultimately resulting
# in one weight update.
BATCH_SIZE: int = 256
# How many times the entire training dataset will be iterated over during the training process
EPOCHS: int = 100
# This controls how often the loss and the metrics for the training process will be printed to the
# console.
EPOCH_STEP: int = 50

# == EVALUATION PARAMETERS ==
# These parameters determine the behavior and the look of the evaluation process, which includes for example
# the generation of visualizations.
LOG_STEP_EVAL: int = 1000

# == EXPERIMENT PARAMETERS ==

__DEBUG__ = True

@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    e.log('starting experiment...')
    e['device'] = tf.device(e.DEVICE)
    e['device'].__enter__()
    
    e.FINAL_UNITS += [e.NUM_TARGETS]

    e.log('starting experiment...')

    e.log('loading dataset...')
    name_data_map, index_data_map = load_visual_graph_dataset(
        path=VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=LOG_STEP_EVAL,
        metadata_contains_index=True
    )
    dataset_indices = list(sorted(index_data_map.keys()))
    dataset_length = len(index_data_map)

    # This callback can be used to apply modifications to each individual graph of the VGD. In this default
    # implementation the edge attributes are replaced with constant edge weights, since the GCN layers are
    # only able to deal with edge weights and not edge attribute vectors.
    @e.hook('modify_g')
    def modify_g(_e, g):
        g['edge_attributes'] = np.array([[1.0] for _ in g['edge_indices']])
        return g

    dataset: t.List[tc.GraphDict] = []
    for index in dataset_indices:
        data = index_data_map[index]
        g = data['metadata']['graph']

        g = e.apply_hook('modify_g', g=g)

        dataset.append(g)

    e.log(f'loaded dataset with {len(dataset)} elements')

    # -- TRAINING THE MODEL --

    @e.hook('create_model', default=True)
    def create_model(_e):
        model = GcnModel(
            units=_e.parameters['UNITS'],
            final_units=_e.parameters['FINAL_UNITS'],
            use_graph_attributes=_e.parameters['USE_GRAPH_ATTRIBUTES'],
            dropout_rate=_e.parameters['DROPOUT_RATE'],
        )
        model.compile(
            loss=mse,
            metrics=mse,
            optimizer=ks.optimizers.Adam(learning_rate=_e.parameters['LEARNING_RATE']),
            run_eagerly=False
        )

        return model


    @e.hook('model_training', default=True)
    def model_training(e, x_train, y_train, x_test, y_test):
        history = model.fit(
            x_train,
            y_train,
            batch_size=e.BATCH_SIZE,
            epochs=e.EPOCHS,
            validation_data=(x_test, y_test),
            validation_freq=1,
            callbacks=LogProgressCallback(
                logger=e.logger,
                epoch_step=e.EPOCH_STEP,
                identifier=f'val_mean_squared_error'
            ),
            verbose=[]
        )

        return history


    for rep in range(REPETITIONS):
        e.log(f'REPETITION ({rep+1}/{REPETITIONS})')

        e.log('creating train test split...')
        train_indices = random.sample(dataset_indices, k=int(TRAIN_RATIO * len(dataset_indices)))
        test_indices = [index for index in dataset_indices if index not in train_indices]
        e[f'train_indices/{rep}'] = train_indices
        e[f'test_indices/{rep}'] = test_indices
        e.log(f'determined {len(train_indices)} train_indices and {len(test_indices)} test indices')

        # This turns the list of graph dicts into the final form which we need for the training of the model:
        # keras RaggedTensors which contain all the graphs.
        x_train, y_train, x_test, y_test = process_graph_dataset(
            dataset,
            train_indices=train_indices,
            test_indices=test_indices,
            use_importances=False,
            use_graph_attributes=e.parameters['USE_GRAPH_ATTRIBUTES'],
        )
        e.log(f'num elements x tuple: {len(x_train)}')
        e.log(f'num elements y tuple: {len(y_train)}')


        model = e.apply_hook('create_model')

        e.log('starting model training...')
        e.apply_hook('before_training')

        history = e.apply_hook(
            'model_training',
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )
        e[f'history/{rep}'] = history
        
        num_params = model.count_params()
        e.log(f'number of model parameters: {num_params}')
        
        @e.hook('model_evaluation', default=True)
        def model_evaluation(e: Experiment,
                             key: str,
                             rep: int,
                             model: t.Any,
                             x: tuple,
                             y: np.ndarray):
            
            e.log(f'evaluating "{key}" set...')
            out_pred = model(x)
            out_true = y[0]
            
            # ~ visualizing the evaluation results
            # This section generates a PDF which contains the R2 scores for all the different target values,
            # calculated using the test set predictions.
            # The PDF also visualizes the regression results using a regression "scatter" plot.

            n_cols = len(TARGET_NAMES)
            n_rows = 1

            # First page is the regressions scatter plots for each of the targets
            fig, rows = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(10 * n_cols, 10), squeeze=False)
            for i, name in enumerate(TARGET_NAMES):
                ax = rows[0][i]

                values_true = []
                values_pred = []
                for value_true, value_pred in zip(out_true, out_pred):
                    if not np.isnan(value_true[i]) and not np.isnan(value_pred[i]):
                        values_true.append(value_true[i])
                        values_pred.append(value_pred[i])

                plot_regression_fit(
                    values_true=values_true,
                    values_pred=values_pred,
                    ax=ax
                )

                r2_value = r2_score(values_true, values_pred)
                mse_value = mse_score(values_true, values_pred)
                rmse_value = np.sqrt(mse_value)
                mae_value = mae_score(values_true, values_pred)
                ax.set_title(f'"{name}"\n'
                                f'r2: {r2_value:.3f} \n'
                                f'mse: {mse_value:.3f} \n'
                                f'rmse: {rmse_value:.3f} \n'
                                f'mae: {mae_value:.3f}')

                # We also want to save it to the experiment store so that we can use that information
                # again later on.
                e[f'r2/{key}/{name}/{rep}'] = float(r2_value)
                e[f'mse/{key}/{name}/{rep}'] = float(mse_value)
                e[f'rmse/{key}/{name}/{rep}'] = float(rmse_value)
                e[f'mae/{key}/{name}/{rep}'] = float(mae_value)

            fig_path = os.path.join(e.path, f'{key}_{rep}.pdf')
            fig.savefig(fig_path)
            plt.close(fig)

        # ~ EVALUATING ON TRAIN SET
        e.apply_hook(
            'model_evaluation',
            key='train',
            rep=rep,
            model=model,
            x=x_train,
            y=y_train,
        )

        # ~ EVALUATING ON TEST SET
        e.apply_hook(
            'model_evaluation',
            key='test',
            rep=rep,
            model=model,
            x=x_test,
            y=y_test,
        )


@experiment.analysis
def analysis(e: Experiment):

    # Creating latex code to display the results in a table
    
    @e.hook('create_latex_table', default=True)
    def create_latex_table(e: Experiment,
                           key: str,
                           names: t.List[str]):
        
        e.log(f'rendering latex table "{key}"...')
        column_names = [
            r'Target Name',
             r'$\text{MAE} \downarrow $',
            r'$\text{MSE} \downarrow $',
            r'$\text{RMSE} \downarrow $',
            r'$R^2 \uparrow $',
        ]
        rows = []
        for name in names:
            row = []

            row.append(name)
            row.append([e[f'mae/{key}/{name}/{rep}'] for rep in range(e.REPETITIONS)])
            row.append([e[f'mse/{key}/{name}/{rep}'] for rep in range(e.REPETITIONS)])
            row.append([e[f'rmse/{key}/{name}/{rep}'] for rep in range(e.REPETITIONS)])
            row.append([e[f'r2/{key}/{name}/{rep}'] for rep in range(e.REPETITIONS)])

            rows.append(row)

        content, table = latex_table(
            column_names=column_names,
            rows=rows,
            list_element_cb=latex_table_element_mean,
            caption=f'Results of {e.REPETITIONS} repetition(s) of ' + r'\textbf{' + e.MODEL_NAME + '}'
        )
        e.commit_raw('table.tex', table)
        pdf_path = os.path.join(e.path, f'table_{key}.pdf')
        render_latex({'content': table}, output_path=pdf_path)
        e.log('rendered latex table')
        
    # ~ Latex table for the training evaluation results
    e.apply_hook(
        'create_latex_table',
        key='train',
        names=e.TARGET_NAMES,
    )
    
    # ~ Latex table for the testing evaluation results
    e.apply_hook(
        'create_latex_table',
        key='test',
        names=e.TARGET_NAMES,
    )


experiment.run_if_main()