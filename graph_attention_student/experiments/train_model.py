"""
This experiment trains one of the models that is defined in this package and then 
evaluates the performance. Note that this experiment does not include any explainability 
aspect, but is rather just focused on training a model normally on either a regression or 
classification task.
"""
import os
import random
import json
import pathlib
import typing as t

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from visual_graph_datasets.web import ensure_dataset
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.config import Config

from graph_attention_student.data import tensors_from_graphs
from graph_attention_student.models.baseline import GCNModel
from graph_attention_student.models.baseline import GATv2Model
from graph_attention_student.models import load_model

PATH: str = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
# These parameters define the source dataset on which the model should be trained on.

# :param VISUAL_GRAPH_DATASET:
#       This parameter defines the dataset to be used for the training of the model.
#       This may be a valid absolute string path pointing to a VGD dataset folder on 
#       the local system. Otherwise this should be the unique string name of a dataset 
#       to be downloaded first from the remote file share location.
VISUAL_GRAPH_DATASET: str = 'logp'
# :param TEST_INDICES_PATH:
#       This is optional. If this is None, then the train / test split is determined randomly.
#       It may also be a valid absolute string path that points towards a json file that 
#       contains a list instance that determines the test indices.
TEST_INDICES_PATH: t.Optional[str] = os.path.join(PATH, 'assets', 'test_indices__logp.json')
# :param NUM_TEST:
#       Only used if no test indices are explictly given. In that case the test set is randomly 
#       chosen as this given number of elements.
NUM_TEST: int = 1000
# :param DATASET_TYPE:
#       Either "regression" or "classification" which determimes how the model functions and how 
#       the results are evaluated.
DATASET_TYPE: str = 'regression'

# == MODEL PARAMETER ==
# These parameters will configure the model architecture itself. They will determine stuff 
# like the layer structure and the numbers of hidden units etc.

# :param CONV_UNITS:
#       This list determines the layer structure of the message passing part of the network. 
#       There will be one layer for each element in this list, the integer list value determines 
#       the number of hidden units of that layer.
CONV_UNITS: t.List[int] = [64, 64, 64]
# :param DENSE_UNITS:
#       This list determines the layer structure of the fully connected prediction MLP.
#       There will be one layer for each element in this list, the integer list value determines
#       the number of hidden units of that layer.
DENSE_UNITS: t.List[int] = [64, 1]

# == TRAINING PARAMETERS ==
# These parameters configure the training process itself. So things like the epochs to train 
# for, the batch size etc.

# :param LEARNING_RATE:
#       The learning rate to be used for the ADAM optimizer during the training.
LEARNING_RATE: float = 1e-3
# :param EPOCHS:
#       The number of epochs to train the network for
EPOCHS: int = 100
# :param BATCH_SIZE:
#       The number of elements to constitute one update step of the network
BATCH_SIZE: int = 100


__DEBUG__ = True

@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment):
    e.log('starting experiment...')
    
    if os.path.exists(e.VISUAL_GRAPH_DATASET):
        dataset_path = e.VISUAL_GRAPH_DATASET
    else:
        config = Config()
        config.load()
        dataset_path = ensure_dataset(
            e.VISUAL_GRAPH_DATASET,
            config=config,
            logger=e.logger,
        )
    
    e['dataset_path'] = dataset_path
    e.log('loading the dataset...')
    reader = VisualGraphDatasetReader(
        path=dataset_path,
        logger=e.logger,
    )
    index_data_map = reader.read()
    indices = list(index_data_map.keys())
    
    e.log('selecting train / test split...')
    if e.TEST_INDICES_PATH is not None:
        e.log(f'loading test indices from file @ {e.TEST_INDICES_PATH}')
        with open(e.TEST_INDICES_PATH, mode='r') as file:
            content = file.read()
            test_indices = json.loads(content)
    else:
        e.log(f'randomly choosing test indices')
        test_indices = random.sample(indices, k=e.NUM_TEST)
    
    train_indices = list(set(indices).difference(set(test_indices)))
    e['test_indices'] = test_indices
    e['train_indices'] = train_indices
    e.log(f'test elements: {len(test_indices)} - train_elements: {len(train_indices)}')
    
    # ~ model training
    # Now that we have the dataset loaded and sorted out we can create and train the model.
    
    @e.hook('train_model')
    def train_model(e: Experiment,
                    index_data_map: dict,
                    train_indices: t.List[int],
                    test_indices: t.List[int]
                    ):
        
        e.log('processing the dataset into tensors for training...')
        graphs_train = [index_data_map[i]['metadata']['graph'] for i in train_indices]
        
        x_train = tensors_from_graphs(graphs_train)
        y_train = np.array([index_data_map[i]['metadata']['target'] for i in train_indices])
        e.log(f'y shape: {y_train.shape}')
        
        e.log('creating the model...')
        model = GCNModel(
            conv_units=e.CONV_UNITS,
            dense_units=e.DENSE_UNITS,
            final_activation='linear',
        )
        
        model.compile(
            optimizer=ks.optimizers.Adam(learning_rate=e.LEARNING_RATE),
            loss=ks.losses.MeanSquaredError(),
        )
        
        e.log('starting model training...')
        model.fit(
            x_train, y_train,
            epochs=e.EPOCHS,
            batch_size=e.BATCH_SIZE,
            verbose=1,
        )
        
        return model
    
    # :hook train_model:
    #       This hook receives the dataset index_data_map and the lists of train and test indices 
    #       as input and is expected to create, train and return the model afterwards.
    model = e.apply_hook(
        'train_model',
        index_data_map=index_data_map,
        test_indices=test_indices,
        train_indices=train_indices,    
    )
    
    # Now after the model has been trained we need to save it to the disk, so that we can use it again 
    # later during the analysis.
    model_path = os.path.join(e.path, 'model')
    model.save(model_path)


@experiment.analysis
def analyis(e: Experiment):
    
    e.log('starting the anaylsis...')
    
    # ~ loading all the assets
    # We load the model and the dataset back into memory
    e.log('loading the model...')
    model_path = os.path.join(e.path, 'model')
    model = load_model(model_path)
    
    e.log('loading the dataset...')
    reader = VisualGraphDatasetReader(
        path=e['dataset_path'],
        logger=e.logger,
    )
    index_data_map = reader.read()
    e.log(f'loaded dataset with {len(index_data_map)} elements')
    graphs_test = [index_data_map[index]['metadata']['graph'] for index in e['test_indices']]
    e.log(f'loaded {len(graphs_test)} graphs for testing')
    
    # ~ save the test indices
    # We might want to use the same test indices for a different experiment and in this case we 
    # will save the experiment conveniently as numpy list 
    e.log('writing the content...')
    test_indices = e['test_indices']
    test_indices_path = os.path.join(e.path, 'test_indices.json')
    with open(test_indices_path, mode='w') as file:
        content = json.dumps(test_indices)
        file.write(content)

    # ~ evaluate model
    # Now we can query the model with the test set and record the predicted outputs so that 
    # we can then later calculate the test set metrics from that
    e.log('query model with test set...')
    predictions = model.predict_graphs(graphs_test)
    # embeddings = model.embedd_graphs(graphs_test)
    
    for index, out in zip(test_indices, predictions):
        data = index_data_map[index]
        graph = data['metadata']['graph']
        
        e[f'out/pred/{index}'] = out
        e[f'out/true/{index}'] = data['metadata']['target']
        
    e.log('evaluating model...')
    if e.DATASET_TYPE == 'regression':
        
        values_true = [e[f'out/true/{i}'] for i in test_indices]
        values_pred = [e[f'out/pred/{i}'] for i in test_indices]
        
        r2_value = r2_score(values_true, values_pred)
        e.log(f' * r2: {r2_value:.3f}')

        mse_value = mean_squared_error(values_true, values_pred)
        e.log(f' * mse: {mse_value:.3f}')
        
        rmse_value = np.sqrt(mse_value)
        e.log(f' * rmse: {rmse_value:.3f}')

    
experiment.run_if_main()