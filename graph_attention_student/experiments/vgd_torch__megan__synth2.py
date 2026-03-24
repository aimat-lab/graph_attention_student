"""
This experiment extends vgd_torch__megan for the training of a self-explaining MEGAN model on a
visual graph dataset.

This experiment specifically implements the synth2 dataset which is a synthetic dataset for
regression tasks. It uses the MoCo-style queue-based contrastive learning for explanation
representation learning.
"""
import typing as t

import numpy as np
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

# == DATASET PARAMETERS ==
# The following parameters determine the dataset and how to handle said dataset.

# :param VISUAL_GRAPH_DATASET:
#       This string may be a valid absolute path to a folder on the local system which
#       contains all the elements of a visual graph dataset. Alternatively this string can be
#       a valid unique identifier of a visual graph dataset which can be downloaded from the main
#       remote file share location.
VISUAL_GRAPH_DATASET: str = 'synth2'
# :param DATASET_TYPE:
#       This string has to determine the type of the dataset in regards to the target values.
#       This can either be "regression" or "classification". This choice influences how the model
#       is trained (loss function) and ultimately how it is evaluated.
DATASET_TYPE: str = 'regression'
# :param TEST_INDICES_PATH:
#       Optionally, this may be an absolute string path to a test indices file.
TEST_INDICES_PATH: t.Optional[str] = None
# :param NUM_TEST:
#       This integer number defines how many elements of the dataset are supposed to be sampled
#       for the unseen test set on which the model will be evaluated. This parameter will be ignored
#       if a test_indices file path is given.
NUM_TEST: int = 1000
# :param USE_BOOTSTRAPPING:
#       This flag determines whether to use bootstrapping with the training elements of the dataset.
USE_BOOTSTRAPPING: bool = False
# :param NUM_EXAMPLES:
#       This integer determines how many elements to sample from the test set elements to act as
#       examples for the evaluation process.
NUM_EXAMPLES: int = 25
# :param TARGET_NAMES:
#       This dictionary structure can be used to define the human readable names for the various
#       target values that are part of the dataset.
TARGET_NAMES: t.Dict[int, str] = {
    0: 'target'
}

# == MODEL PARAMETERS ==
# The following parameters configure the model architecture.

# :param UNITS:
#       This list determines the layer structure of the model's graph encoder part.
UNITS: t.List[int] = [128, 128, 128]
# :param HIDDEN_UNITS:
#       This integer value determines the number of hidden units in the model's graph attention layer's
#       transformative dense networks.
HIDDEN_UNITS: int = 128
# :param IMPORTANCE_UNITS:
#       This list determines the layer structure of the importance MLP.
IMPORTANCE_UNITS: t.List[int] = []
# :param PROJECTION_UNITS:
#       This list determines the layer structure of the MLP's that act as the channel-specific projections.
PROJECTION_UNITS: t.List[int] = [64, 128]
# :param FINAL_UNITS:
#       This list determines the layer structure of the model's final prediction MLP.
#       Note that the last value of this list determines the output shape of the entire network and
#       therefore has to match the number of target values given in the dataset.
FINAL_UNITS: t.List[int] = [128, 64, 1]
# :param NUM_CHANNELS:
#       The number of explanation channels for the model.
NUM_CHANNELS: int = 2
# :param IMPORTANCE_FACTOR:
#       This is the coefficient that is used to scale the explanation co-training loss during training.
IMPORTANCE_FACTOR: float = 1.0
# :param IMPORTANCE_FACTOR_WARMUP_EPOCHS:
#       Number of epochs to linearly ramp up the importance factor from 1e-6 to the final value.
IMPORTANCE_FACTOR_WARMUP_EPOCHS: int = 25
# :param SPARSITY_FACTOR:
#       Coefficient for the explanation sparsity loss (Hoyer-Square regularization).
#       Higher values promote sparser explanation masks.
SPARSITY_FACTOR: float = 0.1
# :param IMPORTANCE_OFFSET:
#       This parameter controls the sparsity of the explanation masks. It acts as a multiplier on
#       the importance values before pooling. Higher values result in more sparse explanations
#       (fewer nodes/edges highlighted), lower values result in denser explanations.
IMPORTANCE_OFFSET: float = 2.5
# :param FIDELITY_FACTOR:
#       This parameter controls the coefficient of the explanation fidelity loss during training.
FIDELITY_FACTOR: float = 0.1
# :param NORMALIZE_EMBEDDING:
#       This boolean value determines whether the graph embeddings are normalized to a unit length or not.
NORMALIZE_EMBEDDING: bool = True
# :param ATTENTION_AGGREGATION:
#       This string literal determines the strategy which is used to aggregate the edge attention logits.
ATTENTION_AGGREGATION: str = 'max'
# :param REGRESSION_MARGIN:
#       When converting the regression problem into the negative/positive classification problem for the
#       explanation co-training, this determines the margin for the thresholding.
REGRESSION_MARGIN: t.Optional[float] = +0.0

# ~ MoCo contrastive learning parameters
# :param CONTRASTIVE_FACTOR:
#       This is the factor of the MoCo contrastive representation learning loss.
CONTRASTIVE_FACTOR: float = 1.0
# :param CONTRASTIVE_NOISE:
#       Noise level for augmentations. Non-explained regions receive this value;
#       explained regions receive 1/4 of this value.
CONTRASTIVE_NOISE: float = 0.2
# :param CONTRASTIVE_TEMP:
#       Temperature for InfoNCE loss. Standard MoCo values are 0.07-0.2.
CONTRASTIVE_TEMP: float = 0.1
# :param CONTRASTIVE_BETA:
#       DEPRECATED. Kept for backward compatibility.
CONTRASTIVE_BETA: float = 1.0
# :param CONTRASTIVE_TAU:
#       DEPRECATED. Kept for backward compatibility.
CONTRASTIVE_TAU: float = 0.1
# :param CONTRASTIVE_QUEUE_SIZE:
#       Size of the MoCo negative queue per explanation channel.
CONTRASTIVE_QUEUE_SIZE: int = 4096
# :param CONTRASTIVE_MOMENTUM:
#       Momentum coefficient for EMA update of momentum projection layers.
CONTRASTIVE_MOMENTUM: float = 0.999
# :param CONTRASTIVE_DETACH_IMPORTANCE:
#       Whether to detach importance masks from contrastive gradients.
CONTRASTIVE_DETACH_IMPORTANCE: bool = True
# :param CONTRASTIVE_WARMUP_EPOCHS:
#       Number of epochs to linearly ramp up the contrastive factor from near zero to the target value.
CONTRASTIVE_WARMUP_EPOCHS: int = 25

# :param TRAIN_MVE:
#       This boolean determines whether or not the model should be trained as a mean variance estimator.
TRAIN_MVE: bool = False
# :param MVE_WARMUP_EPOCHS:
#       This integer determines how many epochs the model should be trained normally before switching on
#       the NLL loss to train the variance as well.
MVE_WARMUP_EPOCHS: int = 50

EPOCHS: int = 150
BATCH_SIZE: int = 32
LEARNING_RATE = 1e-5

__DEBUG__ = True
__TESTING__ = False

experiment = Experiment.extend(
    'vgd_torch__megan.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
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


@experiment.testing
def testing(e: Experiment):
    e.log('TESTING MODE')
    e.NUM_EXAMPLE = 10
    e.EPOCHS = 3


experiment.run_if_main()
