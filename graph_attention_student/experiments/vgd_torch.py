"""
This is the base implementation of an experiment which uses a pytorch-based model to train on 
a visual graph dataset. The experiment trains the model for multiple REPETITIONS and reports 
on the evaluation results for the main prediction performance (regression or classification).

The evaluation of this experiment will create plot artifacts for the most important prediction
performance evaluation such as the regression plot (MAE, MSE, R2 metric) for regression tasks 
and a confusion matrix / AUC plot for classification tasks.
Additionally, this method will visualize some example graphs and provide some debugging 
information about the individual node embeddings.

The analysis of this experiment will load the index_data_map of the dataset and all the saved 
model instances from the disk and store them temporarily into the experiment storage so that 
they are accessible for sub-experimenet analyses.
"""
import os
import json
import random
import typing as t
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import pytorch_lightning as pl
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise_distances
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from visual_graph_datasets.config import Config
from visual_graph_datasets.web import ensure_dataset
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.visualization.base import draw_image
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from lightning.pytorch.loggers import CSVLogger

from graph_attention_student.utils import export_metadatas_csv
from graph_attention_student.visualization import plot_regression_fit
from graph_attention_student.torch.model import AbstractGraphModel
from graph_attention_student.torch.megan import Megan
from graph_attention_student.torch.data import data_list_from_graphs

# == DATASET PARAMETERS ==
# The following parameters determine the dataset and how to handle said dataset.

# :param VISUAL_GRAPH_DATASET:
#       This string may be a valid absolute path to a folder on the local system which 
#       contains all the elements of a visual graph dataset. Alternatively this string can be 
#       a valid unique identifier of a visual graph dataset which can be downloaded from the main 
#       remote file share location.
VISUAL_GRAPH_DATASET: str = 'rb_dual_motifs'
# :param DATASET_TYPE:
#       This string has to determine the type of the dataset in regards to the target values. 
#       This can either be "regression" or "classification". This choice influences how the model 
#       is trained (loss function) and ultimately how it is evaluated.
DATASET_TYPE: str = 'regression' # 'classification'
# :param TEST_INDICES_PATH:
#       Optionally, this may be an absolute string path to a 
TEST_INDICES_PATH: t.Optional[str] = None
# :param NUM_TEST:
#       This integer number defines how many elements of the dataset are supposed to be sampled 
#       for the unseen test set on which the model will be evaluated. This parameter will be ignored 
#       if a test_indices file path is given.
NUM_TEST: int = 1000
# :param USE_BOOTSTRAPPING:
#       This flag determines whether to use bootstrapping with the training elements of the dataset.
#       If enabled, the training samples will be subsampled with the possibility of duplicates. This 
#       method can introduce diversity in the input data distribution between different trained models
#       even though they have the same train-test split.
USE_BOOTSTRAPPING: bool = False
# :param NUM_EXAMPLES:
#       This integer determines how many elements to sample from the test set elements to act as 
#       examples for the evaluation process. These examples will be visualized together with their
#       predictions.
NUM_EXAMPLES: int = 25
# :param TARGET_NAMES:
#       This dictionary structure can be used to define the human readable names for the various 
#       target values that are part of the dataset. The keys of this dict have to be integer indices 
#       of the targets in the order as they appear in the dataset. The values are string which will be 
#       used as the names of these targets within the evaluation visualizations and log messages etc.
TARGET_NAMES: t.Dict[int, str] = defaultdict(str)

# == MODEL PARAMETERS ==
# The following parameters configure the model architecture.

# :param UNITS:
#       This list determines the layer structure of the model's graph encoder part. Each element in 
#       this list represents one layer, where the integer value determines the number of hidden units 
#       in that layer of the encoder network.
UNITS: t.List[int] = [32, 32, 32]
# :param FINAL_UNITS:
#       This list determines the layer structure of the model's final prediction MLP. Each element in 
#       this list represents one layer, where the integer value determines the number of hidden units 
#       in that layer of the prediction network.
#       Note that the last value of this list determines the output shape of the entire network and 
#       therefore has to match the number of target values given in the dataset.
FINAL_UNITS: t.List[int] = [32, 1]

# == TRAINING PARAMETERS ==
# These parameters configure the training process itself, such as how many epochs to train 
# for and the batch size of the training

# :param EPOCHS:
#       The integer number of epochs to train the dataset for. Each epoch means that the model is trained 
#       once on the entire training dataset.
EPOCHS: int = 100
# :param BATCH_SIZE:
#       The batch size to use while training. This is the number of elements from the dataset that are 
#       presented to the model at the same time to estimate the gradient direction for the stochastic gradient 
#       descent optimization.
BATCH_SIZE: int = 32
# :param LEARNING_RATE:
#       This float determines the learning rate of the optimizer.
LEARNING_RATE: float = 1e-3

# == EXPERIMENT PARAMETERS ==
# The following parameters configure the overall behavior of the experiment. This includes 
# meta parameters such as the number of repetitions to repeat the training process to obtain 
# a statistical measure of the model performance.

# :param REPETITIONS:
#       The number of independent times that the training of the model is repeated to obtain a statistical 
#       measure of it's performance.
REPETITIONS: int = 1


__DEBUG__ = True
__TESTING__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('target_from_metadata', default=True, replace=False)
def target_from_metadata(e: Experiment,
                            index: int,
                            metadata: dict,
                            **kwargs,
                            ) -> np.ndarray:
    """
    This hooks is called during the loading of the dataset. It receives the metadata dict for a given element 
    from the dataset's index_data_map and returns the numpy array for the ground truth target value vector. 
    
    This default implementation simply returns the "graph_labels" property of the graph dict representation.
    """
    return metadata['graph']['graph_labels']


@experiment.hook('load_dataset', default=True, replace=False)
def load_dataset(e: Experiment,
                 path: str
                 ) -> dict:
    """
    This hooks is being called to load the dataset. It receives the path to the visual graph dataset folder 
    and returns the loaded index_data_map structure.
    
    This default implemenations additionally iterates over all the elements of the dataset and applies the 
    "target_from_metadata" hook, which can be used to supply custom target values for the elements of the 
    dataset.
    """
    # Given the path to a valid visual graph dataset folder, this reader object will properly load the 
    # dataset into a index_data_map structure, whose keys are the unique integer indices of the dataset 
    # and whose values are again dictionary structures that contain all the metadata (including the full 
    # graph represntation) in the "metadata" field and the absolute path to the visualization image in the 
    # "image_path" field.
    reader = VisualGraphDatasetReader(
        path, 
        logger=e.logger, 
        log_step=1000
    )
    index_data_map = reader.read()
    
    for index, data in index_data_map.items():
    
        metadata = data['metadata']
        graph = metadata['graph']
        
        # :hook target_from_metadata:
        #       This hook receives the metadata dict for each individual element of the dataset and is 
        #       supposed to return the corresponding numpy array that represents the target value vector 
        #       for that element of the dataset. The default implementation of this simply returns the 
        #       the graph dict "graph_labels" array, but the hook can be overwritten to apply custom 
        #       transformations on the target values or to use alternative target values.
        graph['graph_labels'] = e.apply_hook(
            'target_from_metadata',
            index=index,
            metadata=metadata,
        )
        
    return index_data_map


@experiment.hook('load_model')
def load_model(e: Experiment,
                path: str,
                **kwargs
                ) -> AbstractGraphModel:
    """
    This hook receives a file path of a model checkpoint as the parameter and is supposed to load 
    the model from that checkpoint and return the model object instance.
    """
    return Megan.load_from_checkpoint(path)


def train_test_split(e: Experiment,
                     rep: int,
                     indices: list[int],
                     index_data_map: dict[int, dict]
                     ) -> tuple[list, list]:
    """
    This hook is supposed to determine the split of the indices into the training and testing set. 
    The function is supposed to return a tuple (train_indices, test_indices) where both are the lists 
    of integer indices for the train and test set respectively.
    
    This default implementation will first check whether a TEST_INDICES_PATH is defined for the 
    dataset. In case a file exists it will load that file as a JSON list and use the integer elements 
    of that list as the test indices. Otherwise it will sample random test elements from the indices
    list (using NUM_TEST elements) and use the rest for training.
    """
    if e.TEST_INDICES_PATH is not None:
        assert os.path.exists(e.TEST_INDICES_PATH), 'test_indices path not valid / does not exist!'
        assert e.TEST_INDICES_PATH.endswith('.json'), 'test_indices must be JSON file!'

        e.log('found existing test_indices file...')
        with open(e.TEST_INDICES_PATH) as file:
            content = file.read()
            test_indices = json.loads(content)
            
    else:
        e.log('sampling random test elements...')
        test_indices = random.sample(indices, k=e.NUM_TEST)
        
    # The train indices can very simply be derived as all those indices which are not already part of 
    # the test set!
    train_indices = list(set(indices).difference(set(test_indices)))
        
    return train_indices, test_indices


@experiment.hook('train_model', default=True)
def train_model(e: Experiment,
                rep: int,
                index_data_map: dict,
                train_indices: t.List[int],
                test_indices: t.List[int],
                **kwargs,
                ) -> t.Tuple[AbstractGraphModel, pl.Trainer]:
    """
    This hook gets called to actually construct and train a model during the main training loop. This hook 
    receives the full dataset in the format of the index_data_map as well as the train test split in the format 
    of the train_indices and test_indices. The hook is supposed to return a tuple (model, trainer) that were 
    used for the training process.
    """
    e.log('preparing data for training...')
    graphs_train = [index_data_map[i]['metadata']['graph'] for i in train_indices]
    graphs_test = [index_data_map[i]['metadata']['graph'] for i in test_indices]
    
    train_loader = DataLoader(data_list_from_graphs(graphs_train), batch_size=e.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(data_list_from_graphs(graphs_test), batch_size=e.BATCH_SIZE, shuffle=False)
    
    e.log('Instantiating Megan model...')
    model = Megan(
        node_dim=e['node_dim'],
        edge_dim=e['edge_dim'],
        units=e.UNITS,
        importance_units=e.IMPORTANCE_UNITS,
        final_units=e.FINAL_UNITS,
        num_channels=2,
        learning_rate=e.LEARNING_RATE,
        prediction_mode=e.DATASET_TYPE,
    )
    
    e.log(f'starting model training with {e.EPOCHS} epochs...')
    logger = CSVLogger(e[f'path/{rep}'], name='logs')
    trainer = pl.Trainer(
        max_epochs=e.EPOCHS,
        logger=logger,
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
    )
    
    return model, trainer

@experiment.hook('evaluate_model', default=False, replace=False)
def evaluate_model(e: Experiment,
                    model: AbstractGraphModel,
                    trainer: pl.Trainer,
                    rep: int,
                    index_data_map: dict,
                    train_indices: t.List[int],
                    test_indices: t.List[int],
                    ) -> None:
    """
    This hook is called during the main training loop AFTER the model has been fully trained to evaluate 
    the performance of the trained model on the test set. The hook receives the trained model as a parameter,
    as well as the repetition index, the full index_data_map dataset and the train and test indices.
    
    This default implementation only implements an evaluation of the model w.r.t. to the main property 
    prediction task. For a regression tasks it calculates R2, MAE metric and visualizes the regression plot.
    For classification tasks, it calculates Accuracy AUC and creates the confusion matrix.
    
    Additionally, this function will create a plot that visualizes the various training plots over the training 
    epochs if a pytorch lightning log can be found in the archive folder.

    It also visualizes the example graphs (with the chosen NUM_EXAMPLES indices from the test set) together 
    with some information about the node embeddings of those graphs into PDF file.
    """
    e.log('evaluating model prediction performance...')
    archive_path = e[f'path/{rep}']

    graphs_test = [index_data_map[i]['metadata']['graph'] for i in test_indices]
    
    example_indices = e[f'example_indices/{rep}']
    metadatas_example = [index_data_map[i]['metadata'] for i in example_indices]
    graphs_example = [metadata['graph'] for metadata in metadatas_example]
    
    # out_true np.ndarray: (B, O)
    out_true = np.array([graph['graph_labels'] for graph in graphs_test])
    # out_pred np.ndarray: (B, O)
    out_pred = model.predict_graphs(graphs_test)
    
    # ~ exporting the test set
    # Here we want to export the test set predictions into an independent CSV file. This is not strictly necessary 
    # as all the predictions will be saved in the experiment storage anyways, but having it directly in a CSV file 
    # makes it easier to communicate / share the results.
    e.log('exporting test set predictions as CSV...')
    metadatas = []
    for index, graph, out in zip(test_indices, graphs_test, out_pred):
        metadata = index_data_map[index]['metadata']
        metadata['graph']['graph_output'] = out
        metadatas.append(metadata)

    csv_path = os.path.join(e[f'path/{rep}'], 'test.csv')
    export_metadatas_csv(metadatas, csv_path)
    
    # ~ task specific metrics
    # In this section we generate the performance metrics and artifacts for models depending on the specific 
    # tasks type because regression and classification tasks need be treated differently.

    if e.DATASET_TYPE == 'regression':
        e.log('regression task...')
        
        fig, rows = plt.subplots(
            ncols=e['output_dim'],
            nrows=1,
            figsize=(e['output_dim'] * 5, 5),
            squeeze=False
        )
        for target_index in range(e['output_dim']):
            
            ax = rows[0][target_index]
            target_name = e.TARGET_NAMES[target_index]
            
            values_true = out_true[:, target_index]
            values_pred = out_pred[:, target_index]
            e[f'out/true/{rep}'] = values_true
            e[f'out/pred/{rep}'] = values_pred
            
            r2_value = r2_score(values_true, values_pred)
            mse_value = mean_squared_error(values_true, values_pred)
            mae_value = mean_absolute_error(values_true, values_pred)
            e[f'r2/{target_index}/{rep}'] = r2_value
            e[f'mae/{target_index}/{rep}'] = mae_value
            
            plot_regression_fit(
                values_true, values_pred,
                ax=ax,
            )
            ax.set_title(f'target {target_index} - {target_name}\n'
                            f'R2: {r2_value:.3f} - MAE: {mae_value:.3f}')
            
            e.log(f' * {target_index}: {target_name}'
                    f' - r2: {r2_value:.3f}'
                    f' - mse: {mse_value:.3f}'
                    f' - mae: {mae_value:.3f}')
            
        fig.savefig(os.path.join(archive_path, 'regression.pdf'))
        fig.savefig(os.path.join(archive_path, 'regression.png'), dpi=200)
        plt.close(fig)
    
    elif e.DATASET_TYPE == 'classification':
        e.log('classification task...')

        num_classes = out_pred.shape[1]
        e.log(f'classification with {num_classes} classes')
        
        # labels_true: (B, )
        labels_true = np.argmax(out_true, axis=-1)
        # labels_pred: (B, )
        labels_pred = np.argmax(out_pred, axis=-1)
        
        acc_value = accuracy_score(labels_true, labels_pred)
        e[f'acc/{rep}'] = acc_value
        e.log(f' * acc: {acc_value:.3f}')
        
        e.log('plotting confusion matrix...')
        
        cm = confusion_matrix(labels_true, labels_pred)
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
        ticklabels = list(e.TARGET_NAMES.values())
        sns.heatmap(
            cm, 
            ax=ax, 
            annot=True, 
            fmt='02d',
            cmap='viridis',
            xticklabels=ticklabels,
            yticklabels=ticklabels,
            linewidths=0,
        )
        fig.savefig(os.path.join(e[f'path/{rep}'], 'confusion_matrix.pdf'))
        plt.close(fig)
        
        # Only if the classification has exactly 2 clases we can calculate additional metrics for 
        # binary classification as well, such as the AUROC score and the F1 metric.
        if num_classes == 2:
            
            f1_value = f1_score(labels_true, labels_pred)
            e[f'f1/{rep}'] = f1_value
            e.log(f' * f1: {f1_value:.3f}')
        
            auc_value = roc_auc_score(out_true[:, 1], out_pred[:, 1])
            e[f'auc/{rep}'] = auc_value
            e.log(f' * auc: {auc_value:.3f}')
            
            e.log('plotting the AUC curve...')
            
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
            ax.set_title('Receiver Operating Curve')
            fpr, tpr, _ = roc_curve(out_true[:, 1], out_pred[:, 1])
            ax.plot(
                fpr, tpr,
                color='orange',
                label=f'AUC: {auc_value:.3f}'
            )
            ax.plot(
                [0, 1], [0, 1],
                color='lightgray',
                zorder=-10,
                label='random'
            )
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()
            fig.savefig(os.path.join(e[f'path/{rep}'], 'auc.pdf'))
            plt.close(fig)
        
    # ~ plotting loss over epochs
    
    logs_path = os.path.join(e[f'path/{rep}'], 'logs', 'version_0', 'metrics.csv')
    if os.path.exists(logs_path):
        e.log('reading the training logs and plotting the loss...')
        df = pd.read_csv(logs_path)
        
        keys = [name for name in df.columns.tolist() if name != 'epoch' and name.endswith('epoch')]
        e.log(f'plotting the following metrics: {keys}')
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
        for key in keys:
            values = df[key].to_numpy()
            values = values[~np.isnan(values)]
            final_value = values[-1]
            
            ax.plot(
                values,
                label=f'{key} ({final_value:.2f})'
            )
        
        ax.legend()
        ax.set_title('Loss over Training')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        fig.savefig(os.path.join(e[f'path/{rep}'], 'loss.pdf'))
        
    # ~ visualizing examples
    # In this section we want to visualize the graph visualizations of the example elements
    
    e.log('visualizing examples...')
    infos_example = model.forward_graphs(graphs_example)
    
    pdf_path = os.path.join(e[f'path/{rep}'], 'example_graphs.pdf')
    with PdfPages(pdf_path) as pdf:
        
        for index, metadata, info in zip(example_indices, metadatas_example, infos_example):
            
            graph = metadata['graph']
            
            fig, rows = plt.subplots(ncols=3, nrows=1, figsize=(15, 5), squeeze=False)
            fig.suptitle(f'index: {index}')

            ax_img = rows[0][0]
            draw_image(ax=ax_img, image_path=index_data_map[index]['image_path'])
            for node_index, (x, y) in zip(graph['node_indices'], graph['node_positions']):
                ax_img.text(x, y, s=str(node_index))
            
            if 'node_embedding' in info:
                ax_node = rows[0][1]
                ax_node.set_title('Node Embedding Pairwise Distance')
                # node_embedding: (V, D)
                node_embedding = info['node_embedding']
                # node_dist: (V, V)
                #node_dist = np.corrcoef(node_embedding)
                node_dist = pairwise_distances(node_embedding, metric='euclidean')
                sns.heatmap(node_dist, ax=ax_node, annot=False, cmap='crest')

                ax_norm = rows[0][2]
                ax_norm.set_title('Node Embedding L1 Norm')
                # node_norm: (V, )
                node_norm = np.mean(np.abs(node_embedding), axis=-1)
                sns.barplot(node_norm, ax=ax_norm)

            pdf.savefig(fig)
            plt.close(fig)


@experiment
def experiment(e: Experiment):
    
    @e.testing
    def testing(e: Experiment):
        e.EPOCHS = 3
        e.REPETITIONS = 1
        e.NUM_EXAMPLES = 5
    
    e.log('starting experiment...')
    
    # ~ loading the dataset

    if os.path.exists(e.VISUAL_GRAPH_DATASET) and os.path.isdir(e.VISUAL_GRAPH_DATASET):
        e.log(f'dataset seems to be a local folder @ {e.VISUAL_GRAPH_DATASET}')
        dataset_path = e.VISUAL_GRAPH_DATASET
        
    else:
        e.log(f'dataset is not a local folder')
        e.log(f'attempting to fetch dataset from remote file share...')
        config = Config()
        config.load()
        dataset_path = ensure_dataset(e.VISUAL_GRAPH_DATASET, config)
        
    # We want to save the actual path to the experiment storage here so that we can access it again later 
    # during the experiment analysis.
    e['dataset_path'] = dataset_path
        
    e.log('loading dataset...')
    # :hook load_dataset:
    #       This hook receives the path to the visual graph dataset as a parameter and is supposed to return 
    #       the loaded index_data_map as a result.
    index_data_map = e.apply_hook(
        'load_dataset',
        path=dataset_path,
    )
    indices = list(index_data_map.keys())
    e.log(f'loaded dataset with {len(index_data_map)} elements')
        
    # To set up the torch model later on, it is necessary to know the shapes of the input feature vectors.
    # specifically the shapes of the node featuers and the edge features. So here we are extracting those 
    # features for the given dataset from one example graph and saving them to the experiment storage, so 
    # we have access to them later on.
    example_graph = index_data_map[indices[0]]['metadata']['graph']
    e['node_dim'] = example_graph['node_attributes'].shape[1]
    e['edge_dim'] = example_graph['edge_attributes'].shape[1]
    e['output_dim'] = example_graph['graph_labels'].shape[0]
    
    # ~ setting up default hooks
    # In the following section all the default hook implementations are defined which will be needed 
    # in the main training loop later on.
    
    # ~ main training loop
    
    e.log('starting trainig loop...')
    for rep in range(e.REPETITIONS):
        
        e.log(f'REP ({rep+1}/{e.REPETITIONS})')
        
        # ~ archive folder for current repetition
        archive_path = os.path.join(e.path, f'rep{rep:02d}')
        os.mkdir(archive_path)
        e[f'path/{rep}'] = archive_path
        
        # ~ train-test split
        # For each repetition of the model training we potentially want to pick a different train-test split.
        # Only if a path to a test_indices file has been given we use that one instead. Otherwise we choose 
        # as many random test elements as defined in NUM_TEST parameter and use the rest for training.
        
        e.log('determining train-test split...')
        
        # :hook train_test_split:
        #       This hook receives the repetition index, the full list of indices of the dataset and the
        #       index_data_map of the dataset. The hook is supposed to return a tuple (train_indices, test_indices)
        #       where both are lists of integer indices for the train and test set respectively.
        train_indices, test_indices = e.apply_hook(
            'train_test_split',
            rep=rep,
            indices=indices,
            index_data_map=index_data_map,
        )
            
        # the training indices is just all the rest of the elements that were NOT chosen as the
        # Despite the conversion from list to set and back, this implementation is still the fastest when 
        # compared to using a list comprehension for example. This is actually a performance bottleneck for 
        # very large datasets in the millions of elements!
        train_indices = list(set(indices).difference(set(test_indices)))
        num_examples = min(e.NUM_EXAMPLES, len(test_indices))
        example_indices = random.sample(test_indices, k=num_examples)
        e.log(f'using {len(test_indices)} test elements and {len(train_indices)} train_elements')
        
        # Bootstrapping is a method to introduce diversity in the input data distribution between different
        # trained models even though they have the same train-test split. This can be useful to estimate the
        # variance of the model performance.
        # When bootstrapping is enabled, we simply sample the training indices with replacement which 
        # introduces duplicates in the training set.
        if e.USE_BOOTSTRAPPING:
            e.log('sub-sampling the training elements for bootstrapping...')    
            train_indices = random.choices(train_indices, k=len(train_indices))
        
        e[f'test_indices/{rep}'] = test_indices
        e[f'train_indices/{rep}'] = train_indices
        e[f'example_indices/{rep}'] = example_indices
        
        # We also want to save the test indices that were chosen as an individual json file as an artifact 
        # so that it will be easy to reproduce the experiment with the same indices in subsequent runs.
        indices_path = os.path.join(archive_path, 'test_indices.json')
        with open(indices_path, mode='w') as file:
            content = json.dumps(test_indices)
            file.write(content)
            
        # ~ training the model
        # After all the models have been set up, the model has to be trained.
        
        # :hook train_model:
        #       This hook receives the full dataset in the form of the index_data_map and the train_indices 
        #       and test_indices lists. The hook is expected to construct the model, train the model according 
        #       to the experiment configuration and then return a tuple (model, trainer) consisting of the 
        #       now trained model instance and the pl.Trainer instance that was used for the training.
        model, trainer = e.apply_hook(
            'train_model',
            rep=rep,
            index_data_map=index_data_map,
            train_indices=train_indices,
            test_indices=test_indices,
        )
        
        # ~ evaluating the model
        # After the model training is completed we can evaluate the model performance on the test set 
        # and generate the corresponding model evaluation artifacts
        # :hook evaluate_model:
        #       This hook receives the trained model, the current repetition index, the index_data_map and 
        #       and the train_indices and test_indices lists. This hooks should implement the evaluation of 
        #       the model on the test set and should generate the evaluation plots in the archive folder.
        #       The evaluation metrics should also be saved to the experiment storage so that they are accessible 
        #       later on in the experiment analysis as well.
        e.apply_hook(
            'evaluate_model',
            model=model,
            trainer=trainer,
            rep=rep,
            index_data_map=index_data_map,
            train_indices=train_indices,
            test_indices=test_indices,
        )
        
        # ~ saving the model
        # After the model has been trained we also want to save a persistent version of the model
        # (= model checkpoint) to the archive folder so that it can be used in the future.
        e.log('saving the model to the disk...')
        model_path = os.path.join(archive_path, 'model.ckpt')
        trainer.save_checkpoint(model_path)
    

@experiment.analysis
def analysis(e: Experiment):
    
    e.log('starting experiment analysis...')
    
    # ~ loading the dataset
    # Here we load the dataset from the disk again. This is easily done by just using the hook implementation
    # of this process.
    
    e.log('loading the dataset...')
    index_data_map = e.apply_hook(
        'load_dataset',
        path=e['dataset_path']
    )
    e['_index_data_map'] = index_data_map
    e.log(f'loaded dataset with {len(index_data_map)} elements')
    
    # ~ updating the paths
    # Here we update the absolute path strings that are stored in the experiment storage because there is the 
    # chance that the archive folder has been moved from its original location which would invalidate the 
    # paths that are still stored in there from the experiment runtime. But within the analysis runtime, the 
    # e.path attribute refers to the current path of the archive folder.
    e.log('updating the paths...')
    for rep in range(e.REPETITIONS):
        e[f'path/{rep}'] = os.path.join(e.path, f'rep{rep:02d}')
    
    # ~ loading the models
    # Here we go through all the repetitions of the experiment and try to load the corresponding models. 
    # This is actually more of a sanity check to make sure that models are actually loadable!
    e.log('attempting to load the models...')
    for rep in range(e.REPETITIONS):
        model_path = os.path.join(e[f'path/{rep}'], 'model.ckpt')
        
        # :hook load_model:
        #       This hook receives the path to the model checkpoint and is supposed to load the corresponding model 
        #       instance from that checkpoint. 
        model = e.apply_hook(
            'load_model',
            path=model_path,
        )
        assert isinstance(model, AbstractGraphModel), 'model is not of correct type!'
        
        # Here we store the the actual model object instance temporarily into the experiment storage as a way 
        # to make them accessible by potential other analysis processes of the sub-experiments so that these 
        # dont indepdently have to repeat the loading process.
        e[f'_model/{rep}'] = model
        
        e.log('querying the model with test graphs...')
        test_indices = e[f'test_indices/{rep}']
        graphs_test = [index_data_map[index]['metadata']['graph'] for index in test_indices]
        
        # We perform a forward model query with the test graphs here so we can briefly test if the model works 
        # correctly based on the outputs of this forward method.
        infos = model.forward_graphs(graphs_test)
        assert isinstance(infos, list)
        assert isinstance(infos[0], dict)
        assert len(infos[0]) != 0


experiment.run_if_main()