"""
This is the base implementation of an experiment which uses a pytorch-based model to train on
data loaded from a CSV file. Unlike the VGD-based experiments, this experiment does not require
a pre-processed visual graph dataset. Instead, it loads data directly from CSV and computes
graph representations on-the-fly using a Processing instance from visual_graph_datasets.

The evaluation of this experiment will create plot artifacts for the most important prediction
performance evaluation such as the regression plot (MAE, MSE, R2 metric) for regression tasks
and a confusion matrix / AUC plot for classification tasks.
"""
import os
import csv
import json
import random
import typing as t
from collections import defaultdict
from typing import Union, Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
from rich.pretty import pprint
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
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.util import dynamic_import
from torch_geometric.loader import DataLoader
from lightning.pytorch.loggers import CSVLogger

from graph_attention_student.utils import export_metadatas_csv
from graph_attention_student.visualization import plot_regression_fit
from graph_attention_student.torch.model import AbstractGraphModel
from graph_attention_student.torch.megan import Megan
from graph_attention_student.torch.data import data_list_from_graphs


# == EXPERIMENT PARAMETERS ==
# The following parameters are for the experiment in general.

# :param IDENTIFIER:
#       A unique string name which will act like a tag and help to later identify which group
#       the experiment belongs to.
IDENTIFIER: str = 'default'
# :param SEED:
#       The random seed that is used to determine the test split of the dataset.
SEED: int = 42


# == CSV PARAMETERS ==
# The following parameters configure how data is loaded from the CSV file.

# :param CSV_FILE_PATH:
#       The absolute path to the CSV file containing the training data.
CSV_FILE_PATH: str = ''
# :param VALUE_COLUMN_NAME:
#       The name of the column in the CSV file that contains the graph representation
#       (e.g., SMILES string for molecules).
VALUE_COLUMN_NAME: str = 'smiles'
# :param TARGET_COLUMN_NAMES:
#       A list of column names that contain the target values for the prediction task.
TARGET_COLUMN_NAMES: List[str] = ['target']
# :param INDEX_COLUMN_NAME:
#       Optional. The name of the column containing unique integer indices for each element.
#       If None, indices will be assigned sequentially starting from 0.
INDEX_COLUMN_NAME: Optional[str] = None
# :param NUM_CLASSES:
#       For classification tasks, the number of classes. If None and DATASET_TYPE is 'classification',
#       the number of classes will be auto-detected from the data.
NUM_CLASSES: Optional[int] = None


# == PROCESSING PARAMETERS ==
# The following parameters configure how graph representations are computed.

# :param PROCESSING_PATH:
#       Optional. The absolute path to a process.py module that contains a 'processing' attribute
#       which is a Processing instance (e.g., MoleculeProcessing). If None, a default
#       MoleculeProcessing instance will be created.
PROCESSING_PATH: Optional[str] = None
# :param IMAGE_WIDTH:
#       The width in pixels for generated visualization images.
IMAGE_WIDTH: int = 1000
# :param IMAGE_HEIGHT:
#       The height in pixels for generated visualization images.
IMAGE_HEIGHT: int = 1000


# == DATASET PARAMETERS ==
# The following parameters determine dataset handling.

# :param DATASET_TYPE:
#       This string has to determine the type of the dataset in regards to the target values.
#       This can either be "regression" or "classification".
DATASET_TYPE: str = 'regression'
# :param TEST_INDICES_PATH:
#       Optionally, this may be an absolute string path to a JSON file containing the specific indices
#       to be used for the test set.
TEST_INDICES_PATH: t.Optional[str] = None
# :param NUM_TEST:
#       The number of elements to use for the test set. Can be an integer or a float (fraction).
NUM_TEST: Union[int, float] = 0.1
# :param NUM_VAL:
#       The number of elements to use for the validation set. Can be an integer or a float (fraction).
NUM_VAL: Union[int, float] = 0.1
# :param NUM_TRAIN:
#       Optional limit on training set size. If None, use all remaining elements.
NUM_TRAIN: Union[None, int, float] = None
# :param USE_BOOTSTRAPPING:
#       Whether to use bootstrapping (sampling with replacement) for training data.
USE_BOOTSTRAPPING: bool = True
# :param NUM_EXAMPLES:
#       Number of test elements to visualize as examples.
NUM_EXAMPLES: int = 25
# :param TARGET_NAMES:
#       Human readable names for target values.
TARGET_NAMES: t.Dict[int, str] = defaultdict(str)
# :param CLASS_OVERSAMPLING:
#       Whether to oversample minority classes for classification tasks.
CLASS_OVERSAMPLING: bool = False
# :param OVERSAMPLING_FACTORS:
#       Optional custom oversampling factors per class.
OVERSAMPLING_FACTORS: Dict[int, float] = None


# == MODEL PARAMETERS ==
# The following parameters configure the model architecture.

# :param UNITS:
#       Layer structure of the model's graph encoder.
UNITS: t.List[int] = [32, 32, 32]
# :param FINAL_UNITS:
#       Layer structure of the final prediction MLP. Last value = number of targets.
FINAL_UNITS: t.List[int] = [32, 1]
# :param IMPORTANCE_UNITS:
#       Layer structure of the importance MLP for MEGAN models.
IMPORTANCE_UNITS: t.List[int] = [32]


# == TRAINING PARAMETERS ==

# :param EPOCHS:
#       Number of epochs to train.
EPOCHS: int = 100
# :param BATCH_SIZE:
#       Batch size for training.
BATCH_SIZE: int = 32
# :param LEARNING_RATE:
#       Learning rate for the optimizer.
LEARNING_RATE: float = 1e-3


# == EXPERIMENT PARAMETERS ==

# :param REPETITIONS:
#       Number of independent training repetitions.
REPETITIONS: int = 1


__DEBUG__ = True
__TESTING__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


def process_targets(row: dict,
                    target_columns: List[str],
                    dataset_type: str,
                    num_classes: Optional[int] = None) -> np.ndarray:
    """
    Process target values from a CSV row based on the dataset type.

    For regression: returns float values directly.
    For classification: converts integer labels to one-hot encoding if needed.

    :param row: Dictionary containing the CSV row data
    :param target_columns: List of column names containing target values
    :param dataset_type: Either 'regression' or 'classification'
    :param num_classes: Number of classes for classification (required if single label column)
    :returns: numpy array of target values
    """
    if dataset_type == 'regression':
        return np.array([float(row[col]) for col in target_columns])

    elif dataset_type == 'classification':
        if len(target_columns) == 1:
            # Single column with integer labels → one-hot
            label = int(row[target_columns[0]])
            one_hot = np.zeros(num_classes)
            one_hot[label] = 1.0
            return one_hot
        else:
            # Multiple columns (already one-hot or probabilities)
            return np.array([float(row[col]) for col in target_columns])

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


@experiment.hook('load_processing', default=True, replace=False)
def load_processing(e: Experiment) -> object:
    """
    Load or create the Processing instance used for converting values to graphs.

    If PROCESSING_PATH is specified, loads the processing instance from that module.
    Otherwise, creates a default MoleculeProcessing instance.
    """
    if e.PROCESSING_PATH is not None and os.path.exists(e.PROCESSING_PATH):
        e.log(f'loading processing from {e.PROCESSING_PATH}...')
        module = dynamic_import(e.PROCESSING_PATH, 'process')
        processing = module.processing
    else:
        e.log('using default MoleculeProcessing...')
        processing = MoleculeProcessing()

    return processing


@experiment.hook('filter_element', default=True, replace=False)
def filter_element(e: Experiment,
                   value: str,
                   row: dict,
                   processing: object) -> bool:
    """
    Filter elements before graph processing.

    Returns True to keep the element, False to skip it.

    Default implementation filters out:
    - SMILES containing "." (mixtures/salts)
    - Molecules with only a single atom

    :param e: The experiment instance
    :param value: The value string (e.g., SMILES) to filter
    :param row: The full CSV row dictionary
    :param processing: The Processing instance being used
    :returns: True to keep the element, False to skip
    """
    # Filter mixtures (contain ".")
    if '.' in value:
        return False

    # Filter single-atom molecules
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            return False
        if mol.GetNumAtoms() <= 1:
            return False
    except Exception:
        # If RDKit fails, let processing.process() handle the error later
        pass

    return True


@experiment.hook('target_from_metadata', default=True, replace=False)
def target_from_metadata(e: Experiment,
                         index: int,
                         metadata: dict,
                         **kwargs,
                         ) -> np.ndarray:
    """
    Extract target values from metadata. Returns the 'graph_labels' from the graph dict.
    """
    return metadata['graph']['graph_labels']


@experiment.hook('load_dataset', default=True, replace=False)
def load_dataset(e: Experiment, path: str) -> dict:
    """
    Load dataset from CSV file and convert to index_data_map format.

    This hook reads the CSV file, converts each row's value (e.g., SMILES) to a graph
    representation using the Processing instance, and creates a VGD-compatible
    index_data_map structure.
    """
    e.log(f'loading dataset from CSV: {path}')

    # Load the processing instance
    processing = e.apply_hook('load_processing')
    e['_processing'] = processing

    # For classification with single label column, we need to determine num_classes first
    num_classes = e.NUM_CLASSES
    if e.DATASET_TYPE == 'classification' and len(e.TARGET_COLUMN_NAMES) == 1 and num_classes is None:
        e.log('auto-detecting number of classes...')
        with open(path) as file:
            reader = csv.DictReader(file)
            labels = set()
            for row in reader:
                labels.add(int(row[e.TARGET_COLUMN_NAMES[0]]))
            num_classes = len(labels)
            e.log(f'detected {num_classes} classes')
        e['num_classes'] = num_classes

    # Read CSV and convert to graphs
    index_data_map = {}
    skipped = 0
    filtered = 0

    with open(path) as file:
        reader = csv.DictReader(file)
        for c, row in enumerate(reader):
            # Determine index
            if e.INDEX_COLUMN_NAME is not None:
                index = int(row[e.INDEX_COLUMN_NAME])
            else:
                index = c

            # Get the value to convert (e.g., SMILES)
            value = row[e.VALUE_COLUMN_NAME]

            # Apply filter hook BEFORE processing
            keep = e.apply_hook(
                'filter_element',
                value=value,
                row=row,
                processing=processing,
            )
            if not keep:
                filtered += 1
                continue

            try:
                # Convert to graph representation
                graph = processing.process(value)

                # Process target values
                targets = process_targets(
                    row,
                    e.TARGET_COLUMN_NAMES,
                    e.DATASET_TYPE,
                    num_classes
                )
                graph['graph_labels'] = targets

                # Store in VGD-compatible format
                index_data_map[index] = {
                    'metadata': {
                        'index': index,
                        'graph': graph,
                        'value': value,  # Original representation (SMILES)
                    },
                    'image_path': None,  # Will be computed on-demand
                }

            except Exception as exc:
                e.log(f' ! Error processing value "{value}": {exc}')
                skipped += 1
                continue

    e.log(f'loaded {len(index_data_map)} elements, filtered {filtered}, skipped {skipped}')
    return index_data_map


@experiment.hook('get_visualization', default=True, replace=False)
def get_visualization(e: Experiment,
                      index: int,
                      metadata: dict) -> Tuple[str, np.ndarray]:
    """
    Generate or retrieve cached visualization for a graph element.

    Creates visualization on first access and caches it to the archive folder.
    Returns the image path and node positions for explanation overlays.
    """
    viz_dir = os.path.join(e.path, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    image_path = os.path.join(viz_dir, f'{index}.png')

    if not os.path.exists(image_path):
        processing = e['_processing']
        value = metadata['value']

        fig, node_positions = processing.visualize_as_figure(
            value,
            width=e.IMAGE_WIDTH,
            height=e.IMAGE_HEIGHT
        )
        fig.savefig(image_path)
        plt.close(fig)

        # Cache node positions in metadata
        metadata['graph']['node_positions'] = node_positions

    return image_path, metadata['graph'].get('node_positions')


@experiment.hook('load_model')
def load_model(e: Experiment,
               path: str,
               **kwargs
               ) -> AbstractGraphModel:
    """
    Load a model from a checkpoint file.
    """
    return Megan.load_from_checkpoint(path)


@experiment.hook('train_test_split', default=False, replace=False)
def train_test_split(e: Experiment,
                     indices: List[int],
                     index_data_map: Dict[int, dict]
                     ) -> Tuple[List, List, List]:
    """
    Split indices into training, validation, and test sets.
    """
    # ~ test indices
    if e.TEST_INDICES_PATH is not None:
        assert os.path.exists(e.TEST_INDICES_PATH), 'test_indices path not valid!'
        assert e.TEST_INDICES_PATH.endswith('.json'), 'test_indices must be JSON file!'

        e.log('found existing test_indices file...')
        with open(e.TEST_INDICES_PATH) as file:
            content = file.read()
            test_indices = json.loads(content)
    else:
        random.seed(e.SEED)
        num_test = e.NUM_TEST
        if isinstance(e.NUM_TEST, float):
            num_test = int(e.NUM_TEST * len(indices))

        e.log(f'sampling {num_test} random test elements...')
        test_indices = random.sample(indices, k=num_test)
        pprint(test_indices, max_length=10)

    indices = list(set(indices) - set(test_indices))

    # ~ validation indices
    random.seed(e.SEED)
    num_val = e.NUM_VAL
    if isinstance(e.NUM_VAL, float):
        num_val = int(e.NUM_VAL * len(indices))

    e.log(f'sampling {num_val} random validation elements...')
    val_indices = random.sample(indices, k=num_val)
    pprint(val_indices, max_length=10)
    indices = list(set(indices) - set(val_indices))

    # ~ train indices
    if e.NUM_TRAIN is not None:
        num_train = e.NUM_TRAIN
        if isinstance(e.NUM_TRAIN, float):
            num_train = int(e.NUM_TRAIN * len(indices))

        e.log(f'sampling {num_train} random training elements...')
        random.seed()
        train_indices = random.sample(indices, k=num_train)
    else:
        train_indices = indices

    return train_indices, val_indices, test_indices


@experiment.hook('filter_train_indices', default=False, replace=False)
def filter_train_indices(e: Experiment,
                         train_indices: List[int],
                         index_data_map: Dict[int, dict]
                         ) -> List[int]:
    """
    Apply bootstrapping and/or class oversampling to training indices.
    """
    e.log(f'filtering {len(train_indices)} training elements...')

    if e.USE_BOOTSTRAPPING:
        e.log('sub-sampling the training elements for bootstrapping...')
        random.seed()
        train_indices = random.choices(train_indices, k=len(train_indices))

    if e.CLASS_OVERSAMPLING or e.OVERSAMPLING_FACTORS:
        e.log('oversampling the training elements to counter class imbalance...')

        class_indices_map = defaultdict(list)
        for index in train_indices:
            graph_labels = index_data_map[index]['metadata']['graph']['graph_labels']
            label = np.argmax(graph_labels)
            class_indices_map[label].append(index)

        num_max = max([len(indices) for indices in class_indices_map.values()])
        train_indices_sampled = []
        for label, indices in class_indices_map.items():
            num = len(indices)
            if e.CLASS_OVERSAMPLING:
                num = num_max
            if e.OVERSAMPLING_FACTORS:
                num = int(e.OVERSAMPLING_FACTORS.get(label, 1) * num)

            e.log(f' * class: {label} - {len(indices)} elements > {num} oversampled')
            indices_ = random.choices(indices, k=num)
            train_indices_sampled += indices_

        train_indices = train_indices_sampled

    return train_indices


@experiment.hook('train_model', default=True)
def train_model(e: Experiment,
                index_data_map: dict,
                train_indices: t.List[int],
                test_indices: t.List[int],
                **kwargs,
                ) -> t.Tuple[AbstractGraphModel, pl.Trainer]:
    """
    Construct and train a MEGAN model.
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
    logger = CSVLogger(e.path, name='logs')
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
                   index_data_map: dict,
                   train_indices: t.List[int],
                   test_indices: t.List[int],
                   ) -> None:
    """
    Evaluate the trained model on the test set and generate visualizations.
    """
    e.log('evaluating model prediction performance...')

    graphs_test = [index_data_map[i]['metadata']['graph'] for i in test_indices]

    example_indices = e['indices/example']
    metadatas_example = [index_data_map[i]['metadata'] for i in example_indices]
    graphs_example = [metadata['graph'] for metadata in metadatas_example]

    # Generate visualizations for example elements on-demand
    for index in example_indices:
        data = index_data_map[index]
        if data['image_path'] is None:
            image_path, node_positions = e.apply_hook(
                'get_visualization',
                index=index,
                metadata=data['metadata']
            )
            data['image_path'] = image_path
            if node_positions is not None:
                data['metadata']['graph']['node_positions'] = node_positions

    # out_true np.ndarray: (B, O)
    out_true = np.array([graph['graph_labels'] for graph in graphs_test])
    # out_pred np.ndarray: (B, O)
    out_pred = model.predict_graphs(graphs_test)

    # ~ exporting the test set
    e.log('exporting test set predictions as CSV...')
    metadatas = []
    for index, graph, out in zip(test_indices, graphs_test, out_pred):
        metadata = index_data_map[index]['metadata']
        metadata['graph']['graph_output'] = out
        metadatas.append(metadata)

    csv_path = os.path.join(e.path, 'test.csv')
    export_metadatas_csv(metadatas, csv_path)

    # ~ task specific metrics
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
            e[f'out/true'] = values_true
            e[f'out/pred'] = values_pred

            r2_value = r2_score(values_true, values_pred)
            mse_value = mean_squared_error(values_true, values_pred)
            mae_value = mean_absolute_error(values_true, values_pred)
            e[f'metrics/r2/{target_index}'] = r2_value
            e[f'metrics/mae/{target_index}'] = mae_value

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

        fig.savefig(os.path.join(e.path, 'regression.pdf'))
        fig.savefig(os.path.join(e.path, 'regression.png'), dpi=200)
        plt.close(fig)

    elif e.DATASET_TYPE == 'classification':
        e.log('classification task...')

        num_classes = out_pred.shape[1]
        e.log(f'classification with {num_classes} classes')

        labels_true = np.argmax(out_true, axis=-1)
        labels_pred = np.argmax(out_pred, axis=-1)

        acc_value = accuracy_score(labels_true, labels_pred)
        e[f'metrics/acc'] = acc_value
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
        fig.savefig(os.path.join(e.path, 'confusion_matrix.pdf'))
        plt.close(fig)

        if num_classes == 2:
            f1_value = f1_score(labels_true, labels_pred, average='macro')
            e[f'metrics/f1'] = f1_value
            e.log(f' * f1: {f1_value:.3f}')

            auc_value = roc_auc_score(out_true[:, 1], out_pred[:, 1])
            e[f'metrics/auc'] = auc_value
            e.log(f' * auc: {auc_value:.3f}')

            e.log('plotting the AUC curve...')

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
            ax.set_title('Receiver Operating Curve')
            fpr, tpr, _ = roc_curve(out_true[:, 1], out_pred[:, 1])
            ax.plot(fpr, tpr, color='orange', label=f'AUC: {auc_value:.3f}')
            ax.plot([0, 1], [0, 1], color='lightgray', zorder=-10, label='random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()
            fig.savefig(os.path.join(e.path, 'auc.pdf'))
            plt.close(fig)

    # ~ plotting loss over epochs
    logs_path = os.path.join(e.path, 'logs', 'version_0', 'metrics.csv')
    if os.path.exists(logs_path):
        e.log('reading the training logs and plotting the loss...')
        df = pd.read_csv(logs_path)

        keys = [name for name in df.columns.tolist() if name not in ['epoch', 'step']]
        e.log(f'plotting the following metrics: {keys}')

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
        for key in keys:
            values = df[key].to_numpy()
            values = values[~np.isnan(values)]
            final_value = values[-1]

            ax.plot(values, label=f'{key} ({final_value:.2f})')

        ax.legend()
        ax.set_title('Loss over Training')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        fig.savefig(os.path.join(e.path, 'loss.pdf'))

    # ~ visualizing examples
    e.log('visualizing examples...')
    infos_example = model.forward_graphs(graphs_example)

    pdf_path = os.path.join(e.path, 'example_graphs.pdf')
    with PdfPages(pdf_path) as pdf:
        for index, metadata, info in zip(example_indices, metadatas_example, infos_example):
            graph = metadata['graph']

            fig, rows = plt.subplots(ncols=3, nrows=1, figsize=(15, 5), squeeze=False)
            fig.suptitle(f'index: {index}')

            ax_img = rows[0][0]
            image_path = index_data_map[index]['image_path']
            if image_path and os.path.exists(image_path):
                draw_image(ax=ax_img, image_path=image_path)
                if 'node_positions' in graph:
                    for node_index, (x, y) in zip(graph['node_indices'], graph['node_positions']):
                        ax_img.text(x, y, s=str(node_index))

            if 'node_embedding' in info:
                if len(info['node_embedding'].shape) > 2:
                    node_embedding = info['node_embedding'][:, :, 0]
                else:
                    node_embedding = info['node_embedding']

                ax_node = rows[0][1]
                ax_node.set_title('Node Embedding Pairwise Distance')
                node_dist = pairwise_distances(node_embedding, metric='euclidean')
                sns.heatmap(node_dist, ax=ax_node, annot=False, cmap='crest')

                ax_norm = rows[0][2]
                ax_norm.set_title('Node Embedding L1 Norm')
                node_norm = np.mean(np.abs(node_embedding), axis=-1)
                sns.barplot(node_norm, ax=ax_norm)

            pdf.savefig(fig)
            plt.close(fig)


@experiment
def experiment(e: Experiment):

    e.log('starting experiment...')

    # ~ loading the dataset from CSV
    assert os.path.exists(e.CSV_FILE_PATH), f'CSV file not found: {e.CSV_FILE_PATH}'

    e.log('loading dataset from CSV...')
    index_data_map = e.apply_hook(
        'load_dataset',
        path=e.CSV_FILE_PATH,
    )
    indices = list(index_data_map.keys())
    e.log(f'loaded dataset with {len(index_data_map)} elements')

    # Extract feature dimensions from example graph
    example_graph = index_data_map[indices[0]]['metadata']['graph']
    e['node_dim'] = example_graph['node_attributes'].shape[1]
    e['edge_dim'] = example_graph['edge_attributes'].shape[1]
    e['output_dim'] = example_graph['graph_labels'].shape[0]
    e.apply_hook(
        'after_example_graph',
        example_graph=example_graph,
    )

    e.apply_hook(
        'after_dataset',
        index_data_map=index_data_map,
    )
    e['_index_data_map'] = index_data_map

    # ~ train-test split
    e.log('determining train-test split...')
    train_indices, val_indices, test_indices = e.apply_hook(
        'train_test_split',
        indices=indices,
        index_data_map=index_data_map,
    )

    e.log(f' * {len(train_indices)} train')
    e.log(f' * {len(val_indices)} val')
    e.log(f' * {len(test_indices)} test')

    # ~ example indices
    num_examples = min(e.NUM_EXAMPLES, len(test_indices))
    example_indices = random.sample(test_indices, k=num_examples)
    e.log(f'using {len(test_indices)} test elements and {len(train_indices)} train_elements')

    e[f'indices/train'] = train_indices
    e[f'indices/val'] = val_indices
    e[f'indices/test'] = test_indices
    e[f'indices/example'] = example_indices

    train_indices = e.apply_hook(
        'filter_train_indices',
        train_indices=train_indices,
        index_data_map=index_data_map,
        default=train_indices
    )
    e[f'indices/train_filtered'] = train_indices

    e.commit_json('test_indices.json', test_indices)
    e.commit_json('validation_indices.json', val_indices)

    # ~ training the model
    model, trainer = e.apply_hook(
        'train_model',
        index_data_map=index_data_map,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )
    e['_model'] = model

    # ~ evaluating the model
    e.apply_hook(
        'evaluate_model',
        model=model,
        trainer=trainer,
        index_data_map=index_data_map,
        train_indices=train_indices,
        test_indices=test_indices,
    )

    # ~ saving the model
    e.log('saving the model to the disk...')
    model_path = os.path.join(e.path, 'model.ckpt')
    model.save(model_path)


@experiment.analysis
def analysis(e: Experiment):
    e.log('starting experiment analysis...')
    return


experiment.run_if_main()
