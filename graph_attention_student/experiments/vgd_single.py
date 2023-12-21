"""
This is the BASE experiment that implements the 
"""
import os
import sys
import pathlib
import json
import random
import typing as t
from collections import Counter

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
# from pycomex.util import Skippable
# from pycomex.experiment import Experiment
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.visualization.importances import create_importances_pdf
from kgcnn.data.utils import ragged_tensor_from_nested_numpy

import graph_attention_student.typing as tc
from graph_attention_student.data import process_graph_dataset
from graph_attention_student.util import array_normalize, binary_threshold
from graph_attention_student.util import latex_table, latex_table_element_mean
from graph_attention_student.util import render_latex
from graph_attention_student.models.megan import Megan
from graph_attention_student.training import NoLoss
from graph_attention_student.visualization import plot_regression_fit
from graph_attention_student.visualization import plot_regression_value_distribution

mpl.use('Agg')

# == DATASET PARAMETERS ==
# These parameters define the dataset that is to be used for the experiment as well as properties of
# that dataset such as the train-test split for example.

# :param VISUAL_GRAPH_DATASET_PATH:
#       This is the string path to the visual graph dataset folder which will be loaded to train the model.
#       Make sure to adapt this parameter to an existing folder path on your system.
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('/media/ssd/.visual_graph_datasets/datasets/rb_dual_motifs')
# :param DATASET_TYPE:
#       A string which identifies the type of the dataset to be processed here, which at the moment can
#       be either "regression" or "classification".
DATASET_TYPE: str = 'regression'
# :param NUM_TARGETS:
#       This needs to be the number of targets which are part of this dataset aka the number of outputs
#       that will be produced by the model.
NUM_TARGETS: int = 1
# :param TEST_INDICES_PATH:
#       Optionally, this parameter may specify the absolute string path to a JSON file which contains a list of 
#       integer indices. These indices will then be used as the test indices for the experiment. In this case,
#       the rest of the indices will be used as the training indices.
TEST_INDICES_PATH: t.Optional[str] = None
# :param USE_DATASET_SPLIT:
#       Alternatively, 
#       This variable controls if and which existing dataset split to use from the dataset. A visual graph dataset 
#       may contain multiple canonical data splits which are numbered with integer indices. Setting this variable 
#       to such an integer value will select the corresponding split. This value can be set to None, in which case 
#       a new random split will be created for each run of the experiment.
USE_DATASET_SPLIT: t.Optional[int] = None
# :param NUM_TEST:
#       This defines the integer number of elements to be randomly sampled to make up the test set of the 
#       experiment. The rest of the elements will be used as the train set.
NUM_TEST: int = 1000
# :param TRAIN_RATIO:
#       The relative ratio of the dataset that will be used during model training
TRAIN_RATIO: float = 0.8
# :param NUM_EXAMPLES:
#       This is the number of elements from the test set which is going to be used as examples for the
#       visualization of the explanations generated by the model
NUM_EXAMPLES: int = 100
# :param EXAMPLE_INDICES:
#       This may be a list of dataset indices. The corresponding elements will always be part of the example
#       set and be visualized
EXAMPLE_INDICES: t.List[int] = []
# :param IMPORTANCE_CHANNELS:
#       The number of importance channels reflected in the ground truth importances loaded from the dataset
IMPORTANCE_CHANNELS = 1
# :param NODE_IMPORTANCES_KEY:
#       The string key of the graph dict representations to use to retrieve the ground truth node importances
#       different numbers of channels!
NODE_IMPORTANCES_KEY: t.Optional[str] = 'node_importances_1'
# :param EDGE_IMPORTANCES_KEY:
#       The string key of the graph dict representations to use to retrieve the ground truth edge importances
EDGE_IMPORTANCES_KEY: t.Optional[str] = 'edge_importances_1'
# :param USE_NODE_COORDINATES:
#       "node_coordinates" is an optional additional property of vgd graph elements and store the xyz coordinates
#       of a node in 3D space. If this flag is set to true, those coordinates will be appended to the end of the 
#       node attributes array as additonal node features.
USE_NODE_COORDINATES: bool = False
# :param USE_EDGE_ATTRIBUTES:
#       If this flag is False, instead of using the "edge_attributes" of the dataset, those edge attribute feature 
#       arrays are replaced with simple 1.0 edge weights for all edges. This is important because not all GNNs support 
#       the usage of edge attribute vectors!
USE_EDGE_ATTRIBUTES: bool = True
# :param USE_EDGE_LENGTHS:
#       If this flag is true, the "edge_lengths" will be added to the other edge attributes
USE_EDGE_LENGTHS: bool = False


# == MODEL PARAMETERS ==
# These parameters can be used to configure the model

MODEL_NAME: str = 'MOCK'


# == TRAINING PARAMETERS ==
# These parameters control the training process of the models

# :param RANDOM_SEED:
#       This is an integer that determines the behavior of randomized processes in the experiment to make those 
#       semi-deterministic.
RANDOM_SEED: int = 42
# :param REPETITIONS:
#       The number of indepedent repetitions of the training to perform to average the results over at the end
REPETITIONS: int = 1
# :param OPTIMIZER_CB:
#       This has to be a callable with no arguments which returns a new optimizer object. This has to be 
#       a callable and not a just a constant optimizer object because of the tensorflow problems 
#       when trying to use an already compiled optimizer for the training of another model
OPTIMIZER_CB: t.Callable = lambda: ks.optimizers.Adam(learning_rate=1e-3)
# :param BATCH_SIZE:
#       The batch size for the training. This is the number of elements from the training dataset that the 
#       model will get to see at the same time to make each distince parameter update.
BATCH_SIZE: int = 256
# :param EPOCHS:
#       The number of epochs which the model will be trained for. In each epoch the model will get to see the 
#       entire training dataset.
EPOCHS: int = 200
# :param DEVICE:
#       This string defines the device which will be used by tensorflow to perform the training. By default 
#       it is "cpu:0" which indicates the CPU of the system. This is because most user graphics cards do not 
#       have the necessary VRAM to train on even moderately sized graph datasets. A GPU can be selected by 
#       using "gpu:0" etc.
DEVICE: str = 'cpu:0'

# == EVALUATION PARAMETERS ==
# These parameters control the evaluation process, which includes the drawing of visualizations and plots

# :param FIG_SIZE:
#       The base matplotlib figsize used for all the evaluation plots
FIG_SIZE: int = 8
# :param LOG_STEP_EVAL:
#       At how many iterations to print a new log message
LOG_STEP_EVAL: int = 100
# This is the batch size that is used during the evaluation of the test set.
BATCH_SIZE_EVAL: int = 256
# A list which has to have the same number of entries as there are importance channels in this experiment.
# Each string element defines the label that is to be given to that channel in the visualization of
# the example explanations generated by the model.
IMPORTANCE_CHANNEL_LABELS = ['predicted']

# == EXPERIMENT PARAMETERS ==
__DEBUG__ = True
__TESTING__ = True


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):

    if e.__TESTING__:
        e.EPOCHS = 2
        
    # 21.12.23
    # Setting the random seed so that all the random behaviors in this experiment have a replicable 
    # semi-deterministic behavior.
    e.log(f'setting the random seed: {e.RANDOM_SEED}')
    random.seed(e.RANDOM_SEED)

    vgd_name = os.path.basename(VISUAL_GRAPH_DATASET_PATH)
    e.log(f'Starting to train model on vgd "{vgd_name}"')

    # -- DATASET LOADING --

    metadata_map, index_data_map = load_visual_graph_dataset(
        VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=LOG_STEP_EVAL,
        metadata_contains_index=True
    )

    @e.hook('get_graph_labels', default=True)
    def get_graph_labels(e, metadata, graph):
        return metadata['target']

    # As the first step in the processing pipeline we need to get a list of graph dicts, which can then
    # later be turned into the tensors needed for the model.
    dataset_indices = []
    dataset: t.List[tc.GraphDict] = [None for _ in range(max(index_data_map.keys()) + 1)]
    for index in list(sorted(index_data_map.keys())):
        data = index_data_map[index]
        metadata = data['metadata']
        graph: tc.GraphDict = data['metadata']['graph']
        
        # 15.06.23 - If we encounter a target value that we do not want to be part of the dataset, the 
        # hook implementation raises a ValueError which signals to skip that particular element...
        try:
            # :hook get_graph_labels:
            #       this hook receives the metadata dict and the graph dict representation of a single element 
            #       from the VGD dataset and it should return the "graph_labels" numpy array that represents the 
            #       supervised target label for that element.
            graph['graph_labels'] = e.apply_hook(
                'get_graph_labels', 
                metadata=metadata, 
                graph=graph
            )
            
        except ValueError:
            continue
        
        if 'image_path' not in data:
            continue

        if NODE_IMPORTANCES_KEY is not None:
            graph['node_importances'] = graph[NODE_IMPORTANCES_KEY]
        else:
            graph['node_importances'] = np.zeros(shape=(len(graph['node_indices']), IMPORTANCE_CHANNELS))

        if EDGE_IMPORTANCES_KEY is not None:
            graph['edge_importances'] = graph[EDGE_IMPORTANCES_KEY]
        else:
            graph['edge_importances'] = np.zeros(shape=(len(graph['edge_indices']), IMPORTANCE_CHANNELS))

        if USE_NODE_COORDINATES:
            graph['node_attributes'] = np.concatenate(
                [graph['node_attributes'], graph['node_coordinates']],
                axis=-1
            )

        if USE_EDGE_LENGTHS:
            graph['edge_attributes'] = np.concatenate(
                [graph['edge_attributes'], graph['edge_lengths']],
                axis=-1
            )

        if not USE_EDGE_ATTRIBUTES:
            graph['edge_attributes'] = np.ones(shape=(len(graph['edge_indices']), 1))

        # :hook modify_graph:
        #       This hook can be used to modify the graph in the end
        e.apply_hook('modify_graph', index=index, graph=graph)

        dataset[index] = graph
        dataset_indices.append(index)

    dataset_length = len(dataset_indices)
    dataset_indices_set = set(dataset_indices)
    e.log(f'loaded dataset with {dataset_length} elements')

    # Plotting dataset details
    e.log('plotting dataset details...')
    if e.DATASET_TYPE == 'regression':
        fig, rows = plt.subplots(
            ncols=e.NUM_TARGETS,
            nrows=1,
            figsize=(e.NUM_TARGETS * e.FIG_SIZE, e.FIG_SIZE),
            squeeze=False,
        )
        for target_index in range(e.NUM_TARGETS):
            ax = rows[0][target_index]
            plot_regression_value_distribution(
                values=np.array([graph['graph_labels'][target_index] for graph in dataset]),
                ax=ax,
            )

        e.commit_fig('dataset.pdf', fig)

    # Default Hook Implementations

    @e.hook('process_dataset', default=True)
    def process_dataset(e, dataset, train_indices, test_indices):
        e.log(f'processing dataset with {len(dataset)} elements')
        e.log(f'num train indices: {len(train_indices)} - max index: {max(train_indices)}')
        e.log(f'num test indices: {len(test_indices)} - max index: {max(test_indices)}')
        x_train, y_train, x_test, y_test = process_graph_dataset(
            dataset,
            train_indices=train_indices,
            test_indices=test_indices,
        )

        return x_train, y_train, x_test, y_test

    @e.hook('create_model', default=True)
    def create_model(e):
        model = Megan(
            units=[3, 3],
            final_units=[3, 1],
            importance_channels=e.IMPORTANCE_CHANNELS
        )
        model.compile(
            optimizer=e.OPTIMIZER_CB(),
            loss=[ks.losses.MeanSquaredError(), NoLoss(), NoLoss()],
        )
        return model

    @e.hook('fit_model', default=True)
    def fit_model(e, model, x_train, y_train, x_test, y_test):
        hist = model.fit(
            x_train, y_train,
            epochs=e.EPOCHS,
            batch_size=e.BATCH_SIZE,
            verbose=0,
        )
        return hist.history

    @e.hook('query_model', default=True)
    def query_model(e, model, x, y, **kwargs):
        return model(x)

    @e.hook('calculate_fidelity', default=True)
    def calculate_fidelity(e, indices_true, **kwargs):
        return [0 for _ in indices_true]

    # -- MODEL TRAINING --

    for rep in range(REPETITIONS):
        e.log(f'REPETITION ({rep+1}/{REPETITIONS})')
        e['rep'] = rep

        with tf.device(DEVICE):

            e.log(f'determining the dataset train-test split...')
                       
            if e.TEST_INDICES_PATH is not None:
                e.log(f'using external test indices @ {e.TEST_INDICES_PATH}')
                with open(e.TEST_INDICES_PATH, mode='r') as file:
                    content = file.read()
                    test_indices = json.loads(content)
                
            elif e.USE_DATASET_SPLIT is not None:
                e.log(f'creating train test split from dataset...')
                e.log(f'using split from the dataest: {e.USE_DATASET_SPLIT}')
                # A dataset itself has the option to contain one or more *canonical* train test splits. This is 
                # realized as the metadata in every element of the dataset containing a "test_indices" list and 
                # in this list there are multiple "split indices". If this list contains the index of a certain 
                # split then it is a test element for that split.
                test_indices = [index for index, data in index_data_map.items()
                                if USE_DATASET_SPLIT in data['metadata']['test_indices']]

            else:
                e.log(f'creating random train test split...')
                test_indices = random.sample(dataset_indices, k=e.NUM_TEST)

            # In any case, the train indices are all the rest that is left from the elements when we exclude the 
            # test indices - chosen previously by whatever method
            train_indices = list(set(dataset_indices).difference(set(test_indices)))
            
            e.log(f'using {len(train_indices)} training elements and {len(test_indices)} test elements')
            e[f'train_indices/{rep}'] = train_indices
            e[f'test_indices/{rep}'] = test_indices
            
            # :hook filter_indices:
            #       This hook is being called right after the train test split was created for a specific 
            #       experiment repetition. This hook receives as parameters the train indices list, the 
            #       test indices list and the repetition index. The hook function is supposed to return 
            #       a tuple (train_indices, test_indices) consisting of the modified index lists.
            train_indices, test_indices = e.apply_hook(
                'filter_indices',
                train_indices=train_indices,
                test_indices=test_indices,
                rep=rep,
                default=(train_indices, test_indices)
            )
            
            # 21.12.23
            # We also want to export the training and testing indices directly into their own JSON files so that we can 
            # immediately verify them and also we might need those exported files later on as the input to another experiment 
            # which is supposed to operate on the same indices.
            train_indices_path = os.path.join(e.path, f'train_indices__{rep}.json')
            with open(train_indices_path, mode='w') as file:
                content = json.dumps(train_indices)
                file.write(content)
                
            test_indices_path = os.path.join(e.path, f'test_indices__{rep}.json')
            with open(test_indices_path, mode='w') as file:
                content = json.dumps(test_indices)
                file.write(content)
            
            # 20.11.23
            # We'll also export the test indices for the current repetition directly as a json file so that we may easily 
            # access and reuse them later on
            e.log('saving the test_indices as a json file...')
            indices_path = os.path.join(e.path, f'{rep}_test_indices.json')
            with open(indices_path, mode='w') as file:
                content = json.dumps(test_indices)
                file.write(content)

            num_examples = min(NUM_EXAMPLES, len(test_indices)) - len(EXAMPLE_INDICES)
            example_indices = EXAMPLE_INDICES + list(random.sample(test_indices, k=num_examples))
            e.log(f'using {len(example_indices)} as examples')
            e[f'example_indices/{rep}'] = example_indices

            e.log('converting dataset into ragged tensors for training...')
            x_train, y_train, x_test, y_test = e.apply_hook(
                'process_dataset',
                dataset=dataset,
                train_indices=train_indices,
                test_indices=test_indices
            )

            e.log('creating the model...')
            model: ks.models.Model = e.apply_hook('create_model')

            e.log('fitting the model...')
            history: dict = e.apply_hook(
                'fit_model',
                model=model,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
            e[f'history/{rep}'] = history
            e[f'epochs/{rep}'] = list(range(EPOCHS))

            # -- EVALUATING THE MODEL --
            e.log('evaluating test set...')
            out_pred, ni_pred, ei_pred = e.apply_hook(
                'query_model',
                model=model,
                x=x_test,
                y=y_test,
                include_importances=True,
            )
            out_pred = np.array(out_pred.numpy())
            ni_pred = np.array(ni_pred.numpy())
            ei_pred = np.array(ei_pred.numpy())

            out_true = np.array(y_test[0])
            ni_true = np.array(y_test[1].numpy())
            ei_true = np.array(y_test[2].numpy())

            # ~ metrics for the total test set
            e.log('calculating prediction metrics...')
            mse_value = mean_squared_error(out_pred, out_true)
            rmse_value = np.sqrt(mse_value)
            mae_value = mean_absolute_error(out_pred, out_true)
            r2_value = r2_score(out_pred, out_true)
            e[f'mse/{rep}'] = mse_value
            e[f'rmse/{rep}'] = rmse_value
            e[f'mae/{rep}'] = mae_value
            e[f'r2/{rep}'] = r2_value

            e.log('plotting prediction results...')
            if DATASET_TYPE == 'regression':
                fig, rows = plt.subplots(
                    ncols=e.NUM_TARGETS,
                    nrows=1,
                    figsize=(e.NUM_TARGETS * e.FIG_SIZE, e.FIG_SIZE),
                    squeeze=False,
                )
                for target_index in range(e.NUM_TARGETS):
                    ax = rows[0][target_index]
                    
                    values_true = out_true[:, target_index]
                    values_pred = out_pred[:, target_index]
                    # We want to add the numerical evaluation metrics as the title of the plot
                    r2_value = r2_score(values_true, values_pred)
                    mae_value = mean_absolute_error(values_true, values_pred)
                    ax.set_title(f'target {target_index}\n'
                                 f'R2: {r2_value:.3f} - MAE: {mae_value:.3f}')
                    
                    plot_regression_fit(
                        values_true=out_true[:, target_index],
                        values_pred=out_pred[:, target_index],
                        ax=ax,
                    )

                e.commit_fig('regression.pdf', fig)

            e.log('calculating fidelity...')
            fidelities = e.apply_hook(
                'calculate_fidelity',
                model=model,
                dataset=dataset,
                indices_true=test_indices,
                x_true=x_test,
                y_true=y_test,
                out_pred=out_pred,
                ni_pred=ni_pred,
                ei_pred=ei_pred,
            )
            if fidelities:
                for index, fidelity in zip(test_indices, fidelities):
                    e[f'fidelity/{rep}/{index}'] = fidelity

            # ~ calculating explanation metrics
            e.log('calculating explanation metrics...')

            pred_node_importances = []
            pred_edge_importances = []

            true_node_importances = []
            true_edge_importances = []

            for c, index in enumerate(test_indices):
                g = index_data_map[index]['metadata']['graph']
                ni = array_normalize(ni_pred[c])
                ei = array_normalize(ei_pred[c])

                e[f"out/true/{rep}/{index}"] = float(g['graph_labels'][0])
                e[f"out/pred/{rep}/{index}"] = float(out_pred[c][0])

                e[f'ni/pred/{rep}/{index}'] = ni
                e[f'ei/pred/{rep}/{index}'] = ei

                pred_node_importances += ni_pred[c].flatten().tolist()
                pred_edge_importances += ei_pred[c].flatten().tolist()

                true_node_importances += ni_true[c].flatten().tolist()
                true_edge_importances += ei_true[c].flatten().tolist()

                # ~ sparsity of the explanations
                node_sparsity = float(np.mean(binary_threshold(ni, 0.5)))
                edge_sparsity = float(np.mean(binary_threshold(ei, 0.5)))
                e[f'node_sparsity/{rep}/{index}'] = node_sparsity
                e[f'edge_sparsity/{rep}/{index}'] = edge_sparsity

            mean_fidelity = np.mean(list(e[f'fidelity/{rep}'].values()))
            mean_node_sparsity = np.mean(list(e[f'node_sparsity/{rep}'].values()))
            mean_edge_sparsity = np.mean(list(e[f'edge_sparsity/{rep}'].values()))

            # ~ explanation accuracy
            # In the case that there are no ground truth importances (in which case we have set them to
            # be all zeros) this will raise an exception. And in that case we will just return the worst
            # result.
            try:
                node_auc = roc_auc_score(true_node_importances, pred_node_importances)
                edge_auc = roc_auc_score(true_edge_importances, pred_edge_importances)
            except ValueError:
                node_auc = 0.5
                edge_auc = 0.5

            e[f'node_auc/{rep}'] = node_auc
            e[f'edge_auc/{rep}'] = edge_auc

            e.log(f' = mse: {mse_value:.2f}'
                  f' - rmse: {rmse_value:.2f}'
                  f' - mae: {mae_value:.2f}'
                  f' - r2: {r2_value:.2f}'
                  f' - mean fidelity: {mean_fidelity:.2f}'
                  f' - mean node sparsity: {mean_node_sparsity:.2f}'
                  f' - mean edge sparsity: {mean_edge_sparsity:.2f}'
                  f' - node auc: {node_auc:.2f}'
                  f' - edge auc: {edge_auc:.2f}')

            # ~ post-processing
            
            # :hook model_post_process:
            #       This hook can be used to apply additional post processing steps for the model. This could for example 
            #       be additional evaluation steps or even custom model-specific visualizations.
            e.apply_hook(
                'model_post_process', 
                model=model, 
                index_data_map=index_data_map
            )

            # -- VISUALIZATION OF RESULTS --

            # ~ visualizing examples of the graph
            e.log('drawing example visualizations...')
            # First of all we need to query the model using only the elements of the example set
            ni_example_pred = []
            ei_example_pred = []
            for index in example_indices:
                ni_example_pred.append(e[f'ni/pred/{rep}/{index}'])
                ei_example_pred.append(e[f'ei/pred/{rep}/{index}'])

            # Then we can use the already existing method "create_importances_pdf" to create a pdf file
            # which contains the visualizations of the explanations
            pdf_path = os.path.join(e.path, f'examples_{rep:02d}.pdf')
            graph_list = [index_data_map[i]['metadata']['graph'] for i in example_indices]
            image_path_list = [index_data_map[i]['image_path'] for i in example_indices]
            node_positions_list = [g['node_positions'] for g in graph_list]
            create_importances_pdf(
                graph_list=graph_list,
                image_path_list=image_path_list,
                node_positions_list=node_positions_list,
                importances_map={
                    'model': (ni_example_pred, ei_example_pred)
                },
                output_path=pdf_path,
                # importance_channel_labels=IMPORTANCE_CHANNEL_LABELS,
                labels_list=[index_data_map[i]['metadata']['name'] for i in example_indices]
            )

            # 19.04.2021 - Saving the model persistently to the disk now as well optionally
            e.apply_hook(
                'save_model',
                model=model
            )


@experiment.analysis
def analysis(e: Experiment):
    # ~ Loading model
    # As the first step of the analysis we need to load the model in the case that we need 
    # to use that to make some addtional predictions here or something like that.
    model = e.apply_hook('load_model')
    
    # Creating latex code to display the results in a table
    e.log('rendering latex table...')
    column_names = [
        r'Target Name',
        r'$\text{MSE} \downarrow $',
        r'$\text{RMSE} \downarrow $',
        r'$\text{MAE} \downarrow $',
        r'$R^2 \uparrow $',
        r'$\text{Fidelity} \uparrow$',
        r'$\text{Node Sparsity} \downarrow$',
        r'$\text{Edge Sparsity} \downarrow$',
        r'$\text{Node AUC} \uparrow$',
        r'$\text{Edge AUC} \uparrow$',
    ]
    rows = []

    row = []

    row.append('-')
    row.append([e[f'mse/{rep}'] for rep in range(REPETITIONS)])
    row.append([e[f'rmse/{rep}'] for rep in range(REPETITIONS)])
    row.append([e[f'mae/{rep}'] for rep in range(REPETITIONS)])
    row.append([e[f'r2/{rep}'] for rep in range(REPETITIONS)])
    row.append([e[f'fidelity/{rep}/{index}']
                for rep in range(REPETITIONS)
                for index in e[f'test_indices/{rep}']])
    row.append([e[f'node_sparsity/{rep}/{index}']
                for rep in range(REPETITIONS)
                for index in e[f'test_indices/{rep}']])
    row.append([e[f'edge_sparsity/{rep}/{index}']
                for rep in range(REPETITIONS)
                for index in e[f'test_indices/{rep}']])
    row.append([e[f'node_auc/{rep}'] for rep in range(REPETITIONS)])
    row.append([e[f'edge_auc/{rep}'] for rep in range(REPETITIONS)])

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
    e.log('rendered latex table')


experiment.run_if_main()
