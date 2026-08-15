"""
This experiment extends vgd_torch__megan for the training of a self-explaining MEGAN model on a
visual graph dataset.

This experiment specifically implements the training on the rb_dual_motifs_v2 dataset - the
re-generated replacement for rb_dual_motifs. The task is unchanged: a synthetic color graph
regression problem whose target is the sum of four value-determining motifs (blue star -2,
blue ring -1, red ring +1, red star +2), counted once per node-disjoint occurrence.

What changed relative to v1 is the difficulty and the integrity of the collection. v1 leaked
badly - a model could recover 97.9% of the achievable signal from colors and 1-hop
neighbourhoods without ever recognising a motif - and 16% of its elements were exact
duplicates. v2 fills the graphs with *near-misses*: structures that are identical to a motif at
one hop and differ only in whether a single edge closes. There are ~7.8 near-miss sites per
graph against ~2.1 real motifs, so a motif-shaped neighbourhood is nearly four times more
likely to be a decoy. The 1-hop leak drops from 0.972 to 0.163 test R2.

The practical consequence for MEGAN is that the explanation task is genuinely harder: the model
can no longer reach a good prediction by attending to colors alone, so the explanation masks
have to isolate the actual closing edge structure.

Two differences from the v1 experiment that matter:

- v2 is 100,002 elements against v1's 10,000. The schedule is rescaled accordingly (larger
  batch, fewer epochs) so that the total number of gradient steps stays in the same regime as
  the v1 recipe rather than growing tenfold.
- v2 ships only a single-channel ground truth mask (``node_importances_1``), where v1 shipped
  both ``_1`` and the 2-channel ``_2``. The negative/positive ground truth split that would
  line up with MEGAN's two regression channels is therefore not available here, and
  explanation quality is judged by the post-training diagnostic's self-consistency measures
  rather than against the known motifs.
"""
import pathlib
import typing as t

from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path


PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
# The following parameters determine the dataset and how to handle said dataset.

# :param VISUAL_GRAPH_DATASET:
#       This string may be a valid absolute path to a folder on the local system which
#       contains all the elements of a visual graph dataset. Alternatively this string can be
#       a valid unique identifier of a visual graph dataset which can be downloaded from the main
#       remote file share location.
#       The v2 release is not on the remote file share, so this is an absolute local path.
VISUAL_GRAPH_DATASET: str = '/media/ssd/.visual_graph_datasets/datasets/rb_dual_motifs_v2'
# :param DATASET_TYPE:
#       This string has to determine the type of the dataset in regards to the target values.
#       This can either be "regression" or "classification". This choice influences how the model
#       is trained (loss function) and ultimately how it is evaluated.
DATASET_TYPE: str = 'regression'
# :param TEST_INDICES_PATH:
#       Optionally, this may be an absolute string path to a JSON file containing the specific
#       indices to be used for the test set instead of the random test split.
TEST_INDICES_PATH: t.Optional[str] = None
# :param NUM_TEST:
#       This integer number defines how many elements of the dataset are supposed to be sampled
#       for the unseen test set on which the model will be evaluated. This parameter will be ignored
#       if a test_indices file path is given.
#       A plain random split is valid on this dataset: the datasheet states that no two elements
#       share a derivation ancestor and that zero duplicates survive by isomorphism hash, so there
#       is no leakage path between train and test.
#       Fixed counts rather than the 0.1 fractions of the base experiment - on 100k elements a
#       10% split would put 10,000 graphs through the leave-one-out fidelity analysis for no
#       additional statistical value.
NUM_TEST: t.Union[int, float] = 5000
# :param NUM_VAL:
#       The number of elements to sample as the validation set, drawn from what remains after the
#       test set has been taken. This is the set the tuning gate (report.json) is computed on.
NUM_VAL: t.Union[int, float] = 5000
# :param USE_BOOTSTRAPPING:
#       This flag determines whether to use bootstrapping with the training elements of the dataset.
#       If enabled, the training samples will be subsampled with the possibility of duplicates.
USE_BOOTSTRAPPING: bool = False
# :param NUM_EXAMPLES:
#       This integer determines how many elements to sample from the test set elements to act as
#       examples for the evaluation process. These examples will be visualized together with their
#       predictions.
NUM_EXAMPLES: int = 25
# :param TARGET_NAMES:
#       This dictionary structure can be used to define the human readable names for the various
#       target values that are part of the dataset.
TARGET_NAMES: t.Dict[int, str] = {
    0: 'value'
}

# == MODEL PARAMETERS ==
# The following parameters configure the model architecture. These are carried over unchanged
# from the v1 experiment: the task is the same task, so the architecture that solved it should
# be the starting point, and any difference in the results is then attributable to the dataset
# rather than to a retuned model.

# :param UNITS:
#       This list determines the layer structure of the model's graph encoder part. Each element in
#       this list represents one layer, where the integer value determines the number of hidden units
#       in that layer of the encoder network.
UNITS: t.List[int] = [64, 64, 64]
# :param HIDDEN_UNITS:
#       This integer value determines the number of hidden units in the model's graph attention layer's
#       transformative dense networks.
HIDDEN_UNITS: int = 128
# :param IMPORTANCE_UNITS:
#       This list determines the layer structure of the importance MLP which determines the node
#       importance weights from the node embeddings of the graph.
IMPORTANCE_UNITS: t.List[int] = []
# :param PROJECTION_UNITS:
#       This list determines the layer structure of the MLP's that act as the channel-specific
#       projections.
PROJECTION_UNITS: t.List[int] = [64, 128]
# :param FINAL_UNITS:
#       This list determines the layer structure of the model's final prediction MLP. The last value
#       determines the output shape and therefore has to match the number of target values.
FINAL_UNITS: t.List[int] = [64, 1]
# :param NUM_CHANNELS:
#       The number of explanation channels for the model. Two for regression: one for the negative
#       and one for the positive direction relative to the regression reference.
NUM_CHANNELS: int = 2
# :param IMPORTANCE_FACTOR:
#       This is the coefficient that is used to scale the explanation co-training loss during training.
IMPORTANCE_FACTOR: float = 1.0
# :param IMPORTANCE_OFFSET:
#       This parameter controls the sparsity of the explanation masks. It provides the upper limit of
#       how many nodes/edges need to be activated for a channel to be considered as active. The higher
#       this value, the less sparse the explanations will be.
#       Held at the v1 value for the first run. This is the primary lever if the post-training
#       diagnostic reports an explanation problem, in which case it gets swept in 0.2 increments.
IMPORTANCE_OFFSET: float = 1.9
# :param SPARSITY_FACTOR:
#       This is the coefficient that is used to scale the explanation sparsity loss during training.
SPARSITY_FACTOR: float = 1.0
# :param FIDELITY_FACTOR:
#       This parameter controls the coefficient of the explanation fidelity loss during training.
FIDELITY_FACTOR: float = 0.1
# :param REGRESSION_MARGIN:
#       When converting the regression problem into the negative/positive classification problem for
#       the explanation co-training, this determines the margin for the thresholding.
REGRESSION_MARGIN: t.Optional[float] = -0.3
# :param NORMALIZE_EMBEDDING:
#       Whether the graph embeddings are L2 normalized onto the unit sphere.
NORMALIZE_EMBEDDING: bool = True
# :param ATTENTION_AGGREGATION:
#       The strategy used to aggregate the edge attention logits over the message passing layers.
ATTENTION_AGGREGATION: str = 'max'
# :param CONTRASTIVE_FACTOR:
#       The factor of the contrastive representation learning loss of the network.
CONTRASTIVE_FACTOR: float = 1.0
# :param CONTRASTIVE_NOISE:
#       The noise level applied when generating the positive augmentations.
CONTRASTIVE_NOISE: float = 0.1
# :param CONTRASTIVE_TEMP:
#       Temperature hyperparameter of the contrastive learning loss.
CONTRASTIVE_TEMP: float = 1.0
# :param CONTRASTIVE_TAU:
#       Roughly the inverse of the number of expected concepts.
CONTRASTIVE_TAU: float = 0.1
# :param CONTRASTIVE_BETA:
#       The concentration parameter of the hard negative mining.
CONTRASTIVE_BETA: float = 1.0
# :param PREDICTION_FACTOR:
#       The factor by which the main prediction loss is scaled during training.
PREDICTION_FACTOR: float = 1.0

# == TRAINING PARAMETERS ==
# These parameters configure the training process itself.
#
# The v1 recipe was 150 epochs at batch 32 over ~8,000 training elements, which is ~37,500
# gradient steps. v2 has ~90,000 training elements, so the same epoch count would be a tenfold
# increase in compute for no obvious benefit. The schedule below keeps the step count in the
# same regime while taking advantage of the larger dataset:
#
#   90,002 train / batch 64 = ~1,407 steps per epoch  x  40 epochs  =  ~56,000 steps
#
# which is ~1.5x the v1 step count, with every step seeing twice as many graphs and no element
# repeated within an epoch. The learning rate is scaled linearly with the batch size (32 -> 64
# means 1e-5 -> 2e-5) to keep the effective step size comparable.

# :param EPOCHS:
#       The integer number of epochs to train the dataset for.
EPOCHS: int = 40
# :param BATCH_SIZE:
#       The batch size to use while training.
BATCH_SIZE: int = 64
# :param LEARNING_RATE:
#       This float determines the learning rate of the optimizer.
LEARNING_RATE: float = 2e-5

REPETITIONS = 1

__DEBUG__ = True
__TESTING__ = False

experiment = Experiment.extend(
    'vgd_torch__megan.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.testing
def testing(e: Experiment):
    """
    The smoke test only checks that the dataset loads, that the graph tensors have the shape the
    model expects and that a forward/backward pass plus the whole evaluation tail runs without
    crashing. It deliberately subsamples the training set as well as shortening the schedule -
    on 100k elements even a few epochs at the full training size is minutes of pointless compute
    for a test that is looking for exceptions, not for quality.
    """
    e.log('TESTING MODE')
    e.EPOCHS = 3
    e.NUM_TRAIN = 2000
    e.NUM_TEST = 200
    e.NUM_VAL = 200
    e.NUM_EXAMPLES = 10


experiment.run_if_main()
