"""
This experiment extends vgd_torch__megan for the training of a self-explaining MEGAN model on a 
visual graph dataset.

This experiment specifically implements the AqSolDB dataset which is a dataset for predicting the 
experimentally measured values for the water solubility of compounds.
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
VISUAL_GRAPH_DATASET: str = 'aqsoldb'
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
TARGET_NAMES: t.Dict[int, str] = {
    0: 'logS'
}

# == MODEL PARAMETERS ==
# The following parameters configure the model architecture.

# :param UNITS:
#       This list determines the layer structure of the model's graph encoder part. Each element in 
#       this list represents one layer, where the integer value determines the number of hidden units 
#       in that layer of the encoder network.
UNITS: t.List[int] = [128, 128, 128]
# :param HIDDEN_UNITS:
#       This integer value determines the number of hidden units in the model's graph attention layer's
#       transformative dense networks that are used for example to perform the message update and to 
#       derive the attention logits.
HIDDEN_UNITS: int = 128
# :param IMPORTANCE_UNITS:
#       This list determines the layer structure of the importance MLP which determines the node importance 
#       weights from the node embeddings of the graph. 
IMPORTANCE_UNITS: t.List[int] = [ ]
# :param PROJECTION_LAYERS:
#       This list determines the layer structure of the MLP's that act as the channel-specific projections.
#       Each element in this list represents one layer where the integer value determines the number of hidden
#       units in that layer.
PROJECTION_UNITS: t.List[int] = [64, 128]
# :param FINAL_UNITS:
#       This list determines the layer structure of the model's final prediction MLP. Each element in 
#       this list represents one layer, where the integer value determines the number of hidden units 
#       in that layer of the prediction network.
#       Note that the last value of this list determines the output shape of the entire network and 
#       therefore has to match the number of target values given in the dataset.
FINAL_UNITS: t.List[int] = [128, 64, 1]
# :param NUM_CHANNELS:
#       The number of explanation channels for the model.
NUM_CHANNELS: int = 2
# :param IMPORTANCE_FACTOR:
#       This is the coefficient that is used to scale the explanation co-training loss during training.
#       Roughly, the higher this value, the more the model will prioritize the explanations during training.
IMPORTANCE_FACTOR: float = 1.0
# :param IMPORTANCE_OFFSET:
#       This parameter controls the sparsity of the explanation masks even more so than the sparsity factor.
#       It basically provides the upper limit of how many nodes/edges need to be activated for a channel to 
#       be considered as active. The higher this value, the less sparse the explanations will be.
#       Typical values range from 0.2 - 2.0 but also depend on the graph size and the specific problem at 
#       hand. This is a parameter with which one has to experiment until a good trade-off is found!
IMPORTANCE_OFFSET: float = 2.0
# :param FIDELITY_FACTOR:
#       This parameter controls the coefficient of the explanation fidelity loss during training. The higher
#       this value, the more the model will be trained to create explanations that actually influence the
#       model's behavior with a positive fidelity (according to their pre-defined interpretation).
#       If this value is set to 0.0, the explanation fidelity loss is completely disabled (==higher computational
#       efficiency).
FIDELITY_FACTOR: float = 0.1
# :param NORMALIZE_EMBEDDING:
#       This boolean value determines whether the graph embeddings are normalized to a unit length or not.
#       If this is true, the embedding of each individual explanation channel will be L2 normalized such that 
#       it is projected onto the unit sphere.
NORMALIZE_EMBEDDING: bool = False
# :param ATTENTION_AGGREGATION:
#       This string literal determines the strategy which is used to aggregate the edge attention logits over 
#       the various message passing layers in the graph encoder part of the network. This may be one of the 
#       following values: 'sum', 'max', 'min'.
ATTENTION_AGGREGATION: str = 'max'
# :param REGRESSION_MARGIN:
#       When converting the regression problem into the negative/positive classification problem for the 
#       explanation co-training, this determines the margin for the thresholding. Instead of using the regression
#       reference as a hard threshold, values have to be at least this margin value lower/higher than the 
#       regression reference to be considered a class sample.
REGRESSION_MARGIN: t.Optional[float] = -0.1
# :param CONTRASTIVE_FACTOR:
#       This is the factor of the contrastive representation learning loss of the network. If this value is 0 
#       the contrastive repr. learning is completely disabled (increases computational efficiency). The higher 
#       this value the more the contrastive learning will influence the network during training.
CONTRASTIVE_FACTOR: float = 0.00
# :param CONTRASTIVE_NOISE:
#       This float value determines the noise level that is applied when generating the positive augmentations 
#       during the contrastive learning process.
CONTRASTIVE_NOISE: float = 0.1
# :param CONTRASTIVE_TAU:
#       This float value is a hyperparameters of the de-biasing improvement of the contrastive learning loss. 
#       This value should be chosen as roughly the inverse of the number of expected concepts. So as an example 
#       if it is expected that each explanation consists of roughly 10 distinct concepts, this should be chosen 
#       as 1/10 = 0.1
CONTRASTIVE_TAU: float = 0.1
# :param CONTRASTIVE_TEMP:
#       This float value is a hyperparameter that controls the "temperature" of the contrastive learning loss.
#       The higher this value, the more the contrastive learning will be smoothed out. The lower this value,
#       the more the contrastive learning will be focused on the most similar pairs of embeddings.
CONTRASTIVE_TEMP: float = 1.0
# :param CONTRASTIVE_BETA:
#       This is the float value from the paper about the hard negative mining called the concentration 
#       parameter. It determines how much the contrastive loss is focused on the hardest negative samples.
CONTRASTIVE_BETA: float = 1.0
# :param TRAIN_MVE:
#       This boolean determines whether or not the (regression) model should be trained as a mean variance estimator
#       (MVE) model. This would mean that the model predicts the mean and the variance of the target value distribution
#       instead of just the mean. This is useful for regression tasks where the target values are not deterministic
#       but have a certain variance/noise.
TRAIN_MVE: bool = False
# :param MVE_WARMUP_EPOCHS:
#       This integer determines how many epochs the model should be trained normally (MSE loss) before switching on 
#       the NLL loss to train the variance as well. In general it is recommended to fully converge a model on the 
#       normal loss before switching to the NLL loss.
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