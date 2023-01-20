"""
This example shows how to train a MEGAN model for multitask graph regression application based on an
existing "visual graph dataset".

In this example the VGD "organic_solvents" is used. This datasets consists of a few thousand molecules,
which are annotated with 4 target values, representing measurements of solubility within 4 different
organic solvents: water, ethanol, benzene, acetone. For this dataset it is important to note that not
every element is annotated with all values. Most of the time, the target value vectors contain "NaN" fields
which means that the downstream processing pipeline and graph neural network are able to handle this.

**CHANGELOG**

0.1.0 - 16.12.2022 - Initial version

0.2.0 - 30.12.2022 - Additional artifacts have been added: (1) After loading of the dataset a PDF with
information about the target value distribution of the dataset is now created. This is supposed to help
for deciding on appropriate REGRESSION_REFERENCE and REGRESSION_LIMITS values. (2) After the evaluation
step, a PDF with the test set results is created containing a regression plot and the R2 value for each of
the targets.

0.3.0 - 20.01.2023 - (1) Reworked the experiment to use the "hook" feature of the new pycomex version. The
model creating and the training are now default hook implementations which can be overwritten by
sub-experiments. (2) It is now possible to run the experiment for multiple independent REPETITIONS.
(3) The analysis code now generates a latex table which shows the average results for all the individual
targets over all the repetitions.
"""
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import csv
import random
import typing as t
from itertools import product

from pycomex.experiment import Experiment
from pycomex.util import Skippable

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as ks
from matplotlib.backends.backend_pdf import PdfPages
from kgcnn.data.moleculenet import OneHotEncoder
from kgcnn.data.utils import ragged_tensor_from_nested_numpy
from imageio.v2 import imread
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from visual_graph_datasets.util import get_dataset_path
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.visualization.importances import create_importances_pdf

import graph_attention_student.typing as tc
from graph_attention_student.util import latex_table, latex_table_element_mean
from graph_attention_student.util import render_latex
from graph_attention_student.data import process_graph_dataset
from graph_attention_student.training import NoLoss, mse
from graph_attention_student.training import LogProgressCallback
from graph_attention_student.models import Megan
from graph_attention_student.visualization import plot_regression_fit

np.set_printoptions(precision=2)

VERSION = '0.3.0'
SHORT_DESCRIPTION = (
    'This example shows how to train a MEGAN model for multitask graph regression application based on an '
    'existing visual graph dataset.'
)

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
HAS_GRAPH_ATTRIBUTES: bool = False
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
# These paremeters can be used to configure the model

MODEL_NAME: str = 'MEGAN'

# This list defines how many graph convolutional layers to configure the network with. Every element adds
# one layer. The numbers themselves are the hidden units to be used for those layers.
UNITS: t.List[int] = [32, 32, 32, 32]

# The dropout rate which is applied after EVERY graph convolutional layer of the network. Especially for
# large networks (>20 hidden units and multiple importance channels, this dropout proves very useful)
DROPOUT_RATE: float = 0.1

# This is the weight of the additional explanation-only step which is being applied to the network.
# This explanation only step is important to develop actually interpretable explanations. Refer to the
# paper for more details about this.
IMPORTANCE_FACTOR: float = 1.0

# This is another hyperparameter of the explanation only train step. Usually should be between 1 and 10
IMPORTANCE_MULTIPLIER: float = 5

# This is the number of explanation channels that are generated by the model. This is also the number of
# attention heads used in the graph convolutional layers of the network. So to get a "wider" network, this
# parameter can also be increased. However, note that the value can only be != 2 if importance factor is
# set to exactly 0.0!
IMPORTANCE_CHANNELS: int = 2 * NUM_TARGETS

# The coefficient value of the explanation sparsity regularization that is applied to the network. Higher
# values should lead to sparser explanations.
SPARSITY_FACTOR: float = 5.0

# We need to supply the range of possible target values and a reference target value a priori to the
# network for the regression case. The regression limits should be as complete as possible. The reference
# does not have to be in the middle, changing it will influence the development of the explanations quite
# a bit.
REGRESSION_REFERENCE = [
    -3,  # water
    -1,  # benzene
    -1,  # acetone
    -1,  # ethanol
]
REGRESSION_LIMITS = [
    [-10, +2],
    [-4, +1],
    [-4, +1],
    [-4, +1],
]

# At the tail end of the network there is a MLP, which does the final prediction. This list defines the
# depth of that MLP. Each element adds a layer and the values themselves dictate the hidden values.
# In the regression case a final entry with "1" hidden layer as the output layer will implicitly be
# added.
FINAL_UNITS = [100, 50]

# This dropout is applied after EVERY layer in the final MLP. Using this should not be necessary
FINAL_DROPOUT = 0.1

# == TRAINING PARAMETERS ==
# These parameters are hyperparameters that control the training process itself.

REPETITIONS = 5
DEVICE = 'gpu:0'
OPTIMIZER_CB = lambda: ks.optimizers.Nadam(learning_rate=0.001)
BATCH_SIZE = 128
EPOCHS = 200

# == EVALUATION PARAMETERS ==
LOG_STEP = 5
LOG_STEP_EVAL = 1000
METRIC_KEY = 'mean_squared_error'
COLOR_PRIMARY = 'gray'
COLOR_SECONDARY = 'black'

# == EXPERIMENT PARAMETERS ==
DEBUG = True
BASE_PATH = os.getcwd()
NAMESPACE = 'results/vgd_multitask_megan'
with Skippable(), (e := Experiment(base_path=BASE_PATH, namespace=NAMESPACE, glob=globals(), debug_mode=DEBUG)):
    FINAL_UNITS += [NUM_TARGETS]

    e.info('starting experiment for VGD multitask example...')
    e.apply_hook('start_experiment')

    e.info('loading dataset...')
    name_data_map, index_data_map = load_visual_graph_dataset(
        path=VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=LOG_STEP_EVAL,
        metadata_contains_index=True
    )
    dataset_indices: t.List[int] = list(sorted(index_data_map.keys()))
    dataset_indices_set: t.Set[int] = set(dataset_indices)
    dataset_length = len(index_data_map)
    dataset: t.List[tc.GraphDict] = []
    for index in dataset_indices:
        data = index_data_map[index]
        g = data['metadata']['graph']
        # g['graph_labels'] = [v if v is not None else 0 for v in g['graph_labels']]
        if not HAS_IMPORTANCES:
            g['node_importances'] = np.zeros(shape=(len(g['node_indices']), IMPORTANCE_CHANNELS))
            g['edge_importances'] = np.zeros(shape=(len(g['edge_indices']), IMPORTANCE_CHANNELS))

        dataset.append(g)

    # ~ Visualize Dataset Properties
    # Here we create a PDF which illustrates some properties of the dataset relating to the target values.
    pdf_path = os.path.join(e.path, 'dataset.pdf')
    with PdfPages(pdf_path) as pdf:
        # First we are going to create a bar chart which will show for every target value, how many elements
        # are actually annotated with a valid value (which is not None)
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 10))
        fig.suptitle(f'number of elements with valid target values\n'
                     f'{dataset_length} elements total')
        ax.set_xlabel('target property')
        ax.set_ylabel('number of elements')
        ax.set_xticks(range(len(TARGET_NAMES)))
        ax.set_xticklabels(TARGET_NAMES)

        target_values_map: t.Dict[str, t.List[float]] = {}
        for i, name in enumerate(TARGET_NAMES):
            target_values_map[name] = [v for g in dataset if (v := g['graph_labels'][i]) is not None]
            num_valid = len(target_values_map[name])
            ax.bar(
                x=i,
                height=num_valid,
                color=COLOR_PRIMARY
            )

        pdf.savefig(fig)
        plt.close(fig)

        # We plot the value ranges and mean values for all the different target value distributions.
        # This information will be important to decide on the model hyperparameters REGRESSION_REFERENCE
        # and REGRESSION_LIMIT for all of the targets
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 10))
        fig.suptitle(f'target value ranges')
        ax.set_xlabel('target value')
        ax.set_ylabel('target property')
        ax.set_yticks(range(len(TARGET_NAMES)))

        y_labels = []
        for i, name in enumerate(TARGET_NAMES):
            values = target_values_map[name]
            min_value = np.min(values)
            max_value = np.max(values)
            mean_value = np.mean(values)
            ax.barh(
                y=i,
                left=min_value,
                width=abs(max_value - min_value),
                color=COLOR_PRIMARY,
            )
            ax.scatter(
                y=i, x=mean_value,
                color=COLOR_SECONDARY,
                marker='d',
            )
            y_labels.append(f'{name}\n'
                            f'({min_value:.1f} ; {mean_value:.1f} ; {max_value:.1f})')

        ax.set_yticklabels(y_labels)
        pdf.savefig(fig)
        plt.close(fig)

    # -- TRAINING THE MODEL

    @e.hook('create_model', default=True)
    def create_model(_e):

        model = Megan(
            units=_e.parameters['UNITS'],
            final_units=_e.parameters['FINAL_UNITS'],
            use_bias=True,
            importance_channels=_e.parameters['IMPORTANCE_CHANNELS'],
            importance_factor=_e.parameters['IMPORTANCE_FACTOR'],
            sparsity_factor=_e.parameters['SPARSITY_FACTOR'],
            regression_limits=_e.parameters['REGRESSION_LIMITS'],
            regression_reference=_e.parameters['REGRESSION_REFERENCE'],
            dropout_rate=_e.parameters['DROPOUT_RATE'],
            final_dropout_rate=_e.parameters['FINAL_DROPOUT'],
            use_graph_attributes=_e.parameters['HAS_GRAPH_ATTRIBUTES'],
        )

        model.compile(
            loss=[
                mse,
                NoLoss(),
                NoLoss()
            ],
            loss_weights=[
                1,
                0,
                0
            ],
            metrics=[
                mse,
                ks.metrics.MeanAbsoluteError(),
            ],
            optimizer=_e.parameters['OPTIMIZER_CB'](),
            run_eagerly=False
        )

        return model

    @e.hook('model_training', default=True)
    def model_training(_e, model, x_train, y_train, x_test, y_test):
        history = model.fit(
            x_train,
            y_train,
            batch_size=_e.parameters['BATCH_SIZE'],
            epochs=_e.parameters['EPOCHS'],
            validation_data=(x_test, y_test),
            validation_freq=1,
            callbacks=LogProgressCallback(
                logger=_e.logger,
                epoch_step=_e.parameters['LOG_STEP'],
                identifier=f'val_output_1_{METRIC_KEY}'
            ),
            verbose=0
        )

        return history

    for rep in range(REPETITIONS):
        e.info(f'REPETITION ({rep+1}/{REPETITIONS})')

        with tf.device(DEVICE):

            e.info('creating random train-test split...')
            train_indices = random.sample(dataset_indices, k=int(TRAIN_RATIO * len(dataset_indices)))
            train_indices_set = set(train_indices)
            test_indices_set = dataset_indices_set.difference(train_indices_set)
            test_indices = list(test_indices_set)
            e['train_indices'] = train_indices
            e['test_indices'] = test_indices
            e.info(f'chose {len(train_indices)} train indices and {len(test_indices)} test indices')

            # This turns the list of graph dicts into the final form which we need for the training
            # of the model:
            # keras RaggedTensors which contain all the graphs.
            e.info('converting dataset into ragged tensors for model training...')
            x_train, y_train, x_test, y_test = process_graph_dataset(
                dataset,
                train_indices=train_indices,
                test_indices=test_indices,
                use_graph_attributes=HAS_GRAPH_ATTRIBUTES,
            )

            e.info('creating the model...')
            model = e.apply_hook('create_model')

            e.info('starting model training...')
            history = e.apply_hook(
                'model_training',
                model=model,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
            e[f'history/{rep}'] = history

        # -- EVALUATION OF TEST SET --
        e.info('evaluating test set...')

        # first of all we need to pipe the entire test set through the model to get the target value
        # predictions as well as the predicted node and edge importance explanations
        out_pred, ni_pred, ei_pred = model(x_test)
        out_pred = np.array(out_pred.numpy())
        out_true = y_test[0]

        example_indices = random.sample(test_indices, k=int(EXAMPLES_RATIO * len(test_indices)))
        x_example, y_example, _, _ = process_graph_dataset(
            dataset=dataset,
            train_indices=example_indices,
            test_indices=example_indices,
            use_graph_attributes=HAS_GRAPH_ATTRIBUTES,
        )
        out_example, ni_example, ei_example = model(x_example)
        out_example = out_example.numpy()
        ni_example = ni_example.numpy()
        ei_example = ei_example.numpy()

        # -- VISUALIZATION OF RESULTS --
        # The purpose of the following code is to create artifacts which document the results of the model
        # training process in a human-readable way.

        # ~ Visualizing the evaluation results
        # This section generates a PDF which contains the R2 scores for all the different target values,
        # calculated using the test set predictions.
        # The PDF also visualizes the regression results using a regression "scatter" plot.
        pdf_path = os.path.join(e.path, f'training_{rep:02d}.pdf')
        with PdfPages(pdf_path) as pdf:
            n_cols = len(TARGET_NAMES)
            n_rows = 1

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
                mse_value = mean_squared_error(values_true, values_pred)
                rmse_value = np.sqrt(mse_value)
                ax.set_title(f'"{name}"\n'
                             f'r2: {r2_value:.2f}\n'
                             f'mse: {mse_value:.2f}\n'
                             f'rmse: {rmse_value: .2f}')

                e[f'r2/{name}/{rep}'] = r2_value
                e[f'mse/{name}/{rep}'] = mse_value
                e[f'rmse/{name}/{rep}'] = rmse_value

            pdf.savefig(fig)
            plt.close(fig)

        # ~ Visualizing Examples
        # This section will create a PDF which contains the illustrations of the predicted importance
        # explanations of the model for a subset of the test set elements contained within the randomly
        # sampled "example set".
        e.info(f'visualizing {len(example_indices)} example explanations...')
        pdf_path = os.path.join(e.path, f'examples_{rep:02d}.pdf')
        graph_list = [index_data_map[i]['metadata']['graph'] for i in example_indices]
        # "create_importances_pdf" is a rather complex function, which, if given sufficient information about
        # the dataset and the model predictions, will create a PDF file containing the visualized importance
        # explanations using graph visualizations of the underlying VGD at a target location.
        create_importances_pdf(
            graph_list=graph_list,
            image_path_list=[index_data_map[i]['image_path'] for i in example_indices],
            node_positions_list=[index_data_map[i]['metadata']['graph']['node_positions'] for i in example_indices],
            importances_map={
                'model': (ni_example, ei_example)
            },
            labels_list=[f'true {index_data_map[i]["metadata"]["target"]} - pred {out_example[c]}'
                         for c, i in enumerate(example_indices)],
            output_path=pdf_path,
            importance_channel_labels=[f'{name} {direction}'
                                       for name in TARGET_NAMES
                                       for direction in ['negative', 'positive']]
        )


with Skippable(), e.analysis:
    # Creating latex code to display the results in a table
    e.info('rendering latex table...')
    column_names = [
        r'Target Name',
        r'$\text{MSE} \downarrow $',
        r'$\text{RMSE} \downarrow $',
        r'$R^2 \uparrow $',
    ]
    rows = []
    for name in TARGET_NAMES:
        row = []

        row.append(name)
        row.append([e[f'mse/{name}/{rep}'] for rep in range(REPETITIONS)])
        row.append([e[f'rmse/{name}/{rep}'] for rep in range(REPETITIONS)])
        row.append([e[f'r2/{name}/{rep}'] for rep in range(REPETITIONS)])

        rows.append(row)

    content, table = latex_table(
        column_names=column_names,
        rows=rows,
        list_element_cb=latex_table_element_mean,
        caption=f'Results of {REPETITIONS} repetition(s) of ' + r'\textbf{' + MODEL_NAME + '}'
    )
    e.commit_raw('table.tex', table)
    pdf_path = os.path.join(e.path, 'table.pdf')
    render_latex({'content': table}, output_path=pdf_path)
    e.info('rendered latex table')