import os
import pathlib
import typing as t

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from rdkit import Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from graph_attention_student.data import process_graph_dataset

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH: str = os.path.expanduser('/media/ssd/.visual_graph_datasets/datasets/logp')
USE_DATASET_SPLIT: t.Optional[int] = 0
TRAIN_RATIO: float = 0.9
NUM_EXAMPLES: int = 100
NUM_TARGETS: int = 1

NODE_IMPORTANCES_KEY: t.Optional[str] = None
EDGE_IMPORTANCES_KEY: t.Optional[str] = None

USE_NODE_COORDINATES: bool = False
USE_EDGE_LENGTHS: bool = False
USE_EDGE_ATTRIBUTES: bool = True

# :param TARGET_NOISE:
#       This float value determines the magnitude of the additive noise that is added to the 
#       target values during the training to simulate the measurement noise that would be present 
#       in a real-world dataset.
TARGET_NOISE: float = 2

# == MODEL PARAMETERS ==
UNITS = [32, 32, 32]
DROPOUT_RATE = 0.0
EMBEDDING_UNITS = [32, 16]
FINAL_UNITS = [1, ]
REGRESSION_REFERENCE = 2.45
REGRESSION_WEIGHTS = [1.0, 1.0]
IMPORTANCE_CHANNELS: int = 2
IMPORTANCE_FACTOR = 2.0
IMPORTANCE_MULTIPLIER = 0.5
FIDELITY_FACTOR = 0.1
FIDELITY_FUNCS = [
    lambda org, mod: tf.nn.relu(mod - org),
    lambda org, mod: tf.nn.relu(org - mod),
]
SPARSITY_FACTOR = 1.0
CONCAT_HEADS = False

# == TRAINING PARAMETERS ==
BATCH_SIZE = 32
EPOCHS = 100
REPETITIONS = 1
OPTIMIZER_CB = lambda: ks.optimizers.experimental.Adam(
    learning_rate=0.001,
)
DEVICE: str = 'cpu:0'

# == EXPERIMENT PARAMETERS ==
__DEBUG__ = True
__TESTING__ = False

experiment = Experiment.extend(
    'vgd_single__megan2.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('process_dataset', default=True)
def process_dataset(e: Experiment, 
                    dataset: t.List[dict], 
                    train_indices: t.List[int], 
                    test_indices: t.List[int]
                    ):
    
    # ~ adding random noise
    # For this data we do a custom pre-processing step where we add rad
    e.log(f'Adding random noise to the target values of the training data')
    e.log(f' * magnitude uniform noise: {e.TARGET_NOISE}')
    for index in train_indices:
        graph = dataset[index]
        
        if e.TARGET_NOISE > 0:
            target = np.array(graph['graph_labels'])
            target += np.random.normal(
                0,
                e.TARGET_NOISE,
                size=target.shape,
            )
            graph['graph_labels'] = target
    
    # This next section is the default implementation that turns the 
    e.log(f'processing dataset with {len(dataset)} elements')
    e.log(f'num train indices: {len(train_indices)} - max index: {max(train_indices)}')
    e.log(f'num test indices: {len(test_indices)} - max index: {max(test_indices)}')
    x_train, y_train, x_test, y_test = process_graph_dataset(
        dataset,
        train_indices=train_indices,
        test_indices=test_indices,
    )

    return x_train, y_train, x_test, y_test

experiment.run_if_main()
