"""
This is the base implementation of an experiment which uses a pytorch-based model to train on 
a visual graph dataset. The experiment traines the model for multiple REPETITIONS and reports 
on the evaluation results for the main prediction performance (regression or classification).
"""
import os
import json
import random
import typing as t
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from visual_graph_datasets.config import Config
from visual_graph_datasets.web import ensure_dataset
from visual_graph_datasets.data import VisualGraphDatasetReader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_attention_student.visualization import plot_regression_fit
from graph_attention_student.torch.model import AbstractGraphModel
from graph_attention_student.torch.megan import Megan
from graph_attention_student.torch.data import data_list_from_graphs

# == DATASET PARAMETERS ==
# The following parameters determine the dataset and how to handle said dataset.

VISUAL_GRAPH_DATASET: str = 'rb_dual_motifs'

DATASET_TYPE: str = 'regression' # 'classification'

TEST_INDICES_PATH: t.Optional[str] = None
NUM_TEST: int = 1000
NUM_EXAMPLES: int = 100
TARGET_NAMES: t.Dict[int, str] = defaultdict(str)

# == MODEL PARAMETERS ==

UNITS: t.List[int] = [32, 32, 32]
IMPORTANCE_UNITS: t.List[int] = [32, 32]
FINAL_UNITS: t.List[int] = [32, 1]

# == TRAINING PARAMETERS ==
# These parameters configure the training process itself, such as how many epochs to train 
# for and the batch size of the training

EPOCHS: int = 100
BATCH_SIZE: int = 16
LEARNING_RATE: float = 1e-3

# == EXPERIMENT PARAMETERS ==
# The following parameters configure the overall behavior of the experiment.

REPETITIONS: int = 1


__DEBUG__ = True
__TESTING__ = True

@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)
def experiment(e: Experiment):
    
    @e.testing
    def testing(e: Experiment):
        e.EPOCHS = 3
        e.REPETITIONS = 1
    
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
    
    @e.hook('target_from_metadata', default=True, replace=False)
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
    
    @e.hook('load_dataset', default=True, replace=False)
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
            dataset_path, 
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
    
    @e.hook('train_model', default=True)
    def train_model(e: Experiment,
                    index_data_map: dict,
                    train_indices: t.List[int],
                    test_indices: t.List[int],
                    **kwargs,
                    ) -> t.Tuple[AbstractGraphModel, pl.Trainer]:
        """
        Th
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
        )
        
        e.log(f'starting model training with {e.EPOCHS} epochs...')
        trainer = pl.Trainer(
            max_epochs=e.EPOCHS,
        )
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=test_loader,
        )
        
        return model, trainer
    
    @e.hook('evaluate_model', default=True, replace=False)
    def evaluate_model(e: Experiment,
                       model: AbstractGraphModel,
                       rep: int,
                       index_data_map: dict,
                       train_indices: t.List[int],
                       test_indices: t.List[int],
                       ) -> None:
        
        e.log('evaluating model prediction performance...')
    
        graphs_test = [index_data_map[i]['metadata']['graph'] for i in test_indices]
        
        # out_true np.ndarray: (B, O)
        out_true = np.array([graph['graph_labels'] for graph in graphs_test])
        # out_pred np.ndarray: (B, O)
        out_pred = model.predict_graphs(graphs_test)
    
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
                mae_value = mean_absolute_error(values_true, values_pred)
                e[f'r2/{rep}'] = r2_value
                e[f'mae/{rep}'] = mae_value
                
                plot_regression_fit(
                    values_true, values_pred,
                    ax=ax,
                )
                ax.set_title(f'target {target_index} - {target_name}\n'
                             f'R2: {r2_value:.3f} - mae: {mae_value:.3f}')
                
                e.log(f' * {target_index}: {target_name}'
                      f' - r2: {r2_value:.3f}'
                      f' - mae: {mae_value:.3f}')
                
            fig.savefig(os.path.join(archive_path, 'regression.pdf'))
            fig.savefig(os.path.join(archive_path, 'regression.png'), dpi=200)
            plt.close(fig)
        
        elif e.DATASET_TYPE == 'classification':
            e.log('classification task...')
    
    @e.hook('load_model')
    def load_model(e: Experiment,
                   path: str,
                   **kwargs
                   ) -> AbstractGraphModel:
        """
        This hook receives a file path of a model checkpoint as the parameter and is supposed to load 
        the model from that checkpoint and return the model object instance.
        """
        return Megan.load_from_checkpoint(path)
    
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
            
        # the training indices is just all the rest of the elements that were NOT chosen as the
        # Despite the conversion from list to set and back, this implementation is still the fastest when 
        # compared to using a list comprehension for example. This is actually a performance bottleneck for 
        # very large datasets in the millions of elements!
        train_indices = list(set(indices).difference(set(test_indices)))
        e.log(f'using {len(test_indices)} test elements and {len(train_indices)} train_elements')
        
        e[f'test_indices/{rep}'] = test_indices
        e[f'train_indices/{rep}'] = train_indices
        
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
    e.log(f'loaded dataset with {len(index_data_map)} elements')
    
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
        
    

experiment.run_if_main()