"""
TEMPLATE: train a MEGAN model on a new molecular dataset (a CSV of SMILES + target).

HOW TO USE THIS TEMPLATE
------------------------
1. Copy this file into  graph_attention_student/experiments/
   It MUST live next to its base experiment `train_model__megan.py` — pycomex resolves
   the base by filename relative to THIS file's own folder. A copy anywhere else cannot
   find `train_model__megan.py` and will fail to start.
2. Rename it to something dataset-specific, e.g.  train_model__megan__solubility.py
   The filename (minus .py) becomes the experiment namespace and the archive folder name.
3. Fill in every `# TODO` below.
4. Smoke test:  set  __TESTING__ = True  then run
       python graph_attention_student/experiments/train_model__megan__<dataset>.py
   The @experiment.testing hook drops it to 3 epochs and only verifies that the data
   loads and the wiring is correct (it will NOT produce good explanations).
5. Full run:    set  __TESTING__ = False  then run the same command.
   (GPU strongly preferred. If you hit "CUDA error: no kernel image is available",
    prefix the command with  CUDA_VISIBLE_DEVICES=""  to fall back to CPU.)
6. Read the diagnostic self-check:
       graph_attention_student/experiments/results/train_model__megan__<dataset>/debug/report.json
   plus scorecard.png / examples.png / fidelity.png in the same folder.
"""
import typing as t
from typing import List, Optional

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace


# == DATASET (CSV) PARAMETERS ==
# :param CSV_FILE_PATH: absolute path to the CSV file with the training data.
CSV_FILE_PATH: str = '/absolute/path/to/your_dataset.csv'  # TODO
# :param VALUE_COLUMN_NAME: name of the column that holds the SMILES strings.
VALUE_COLUMN_NAME: str = 'smiles'  # TODO
# :param TARGET_COLUMN_NAMES: name(s) of the target column(s).
#   - regression: one column name of continuous values.
#   - classification: a single column of integer class labels (0, 1, 2, ...).
TARGET_COLUMN_NAMES: List[str] = ['target']  # TODO
# :param DATASET_TYPE: 'regression' or 'classification'.
DATASET_TYPE: str = 'regression'  # TODO
# :param NUM_CLASSES: number of classes (classification only; else leave None).
NUM_CLASSES: Optional[int] = None  # TODO for classification
# :param NUM_TEST / NUM_VAL: fraction (0-1) or absolute count held out for test / validation.
NUM_TEST: t.Union[int, float] = 0.1
NUM_VAL: t.Union[int, float] = 0.1
# :param TARGET_NAMES: human-readable name per target index (shown in plots).
TARGET_NAMES: t.Dict[int, str] = {0: 'target'}  # TODO


# == EXPLANATION CHANNELS ==
# :param NUM_CHANNELS:
#   - regression:     MUST be 2  (channel 0 = "negative", channel 1 = "positive").
#   - classification: MUST equal the number of classes.
NUM_CHANNELS: int = 2  # TODO for classification
# :param CHANNEL_INFOS: what each channel MEANS for this task (used in plots + human report).
#   For regression keep the negative/positive convention; rename to be task-specific if helpful
#   (e.g. "decreases solubility" / "increases solubility").
CHANNEL_INFOS: dict = {  # TODO: give the channels task-specific names
    0: {'name': 'negative', 'color': 'skyblue'},
    1: {'name': 'positive', 'color': 'coral'},
}


# == MODEL ARCHITECTURE ==
# :param UNITS: sizes of the graph-encoder layers.
UNITS: List[int] = [64, 64, 64]
# :param FINAL_UNITS: final MLP. LAST value MUST equal the number of targets (regression)
#   or the number of classes (classification).
FINAL_UNITS: List[int] = [64, 32, 1]  # TODO: last value = #targets or #classes


# == EXPLANATION KNOBS ==
# Start from these defaults. Tune in response to a FAILED diagnostic self-check;
# see references/knobs-and-troubleshooting.md for the failure -> knob mapping.
#
# :param IMPORTANCE_OFFSET: THE PRIMARY explanation lever. Controls how much of each graph the
#   masks cover and interacts with accuracy/separation. When explanations are unsatisfactory,
#   SWEEP THIS FIRST in increments of 0.2 (useful range ~0.2 - 2.0). Higher -> more coverage;
#   lower -> more focused/sparse.
IMPORTANCE_OFFSET: float = 0.8
# :param REGRESSION_MARGIN: (regression only) SECONDARY lever, in units of the target's std.
#   Values > 0 exclude samples within mean +/- margin*std from the co-training loss, so only
#   clearly negative/positive molecules shape the masks (cleaner separation). Keep 0.0; raise to
#   0.1 - 0.2 in harder cases. NEGATIVE values currently have no effect.
REGRESSION_MARGIN: float = 0.0
# :param IMPORTANCE_FACTOR: weight of the explanation co-training loss (higher = explanations
#   prioritized more during training).
IMPORTANCE_FACTOR: float = 1.0
# :param ATTENTION_AGGREGATION: how edge attention is aggregated across layers.
#   'max' | 'min' | 'sum' | 'mean'. 'min' tends to give the sparsest masks.
ATTENTION_AGGREGATION: str = 'max'
# :param FIDELITY_FACTOR: 0.0 disables the fidelity loss. Set > 0 (e.g. 0.1) as a SECONDARY way
#   to enforce correct fidelity sign/magnitude if fidelity_sign / fidelity_magnitude fail.
FIDELITY_FACTOR: float = 0.0
# :param SPARSITY_FACTOR: extra Hoyer sparsity regularization (marked DEPRECATED in the base, but
#   still functional). Prefer IMPORTANCE_OFFSET for sparsity; leave at default unless needed.
SPARSITY_FACTOR: float = 1.0
# :param REGRESSION_REFERENCE: auto-managed — the model overwrites it with the running target mean
#   during training, and it only affects output centering. NOT a tuning knob; leave at default.
REGRESSION_REFERENCE: Optional[float] = 0.0


# == CLASSIFICATION ONLY (leave defaults for regression) ==
# :param CLASS_OVERSAMPLING: oversample minority classes toward balance (helps skewed datasets).
CLASS_OVERSAMPLING: bool = False
# :param OVERSAMPLING_FACTORS: optional manual per-class oversampling, e.g. {0: 1, 1: 3}.
OVERSAMPLING_FACTORS: Optional[dict] = None
# :param LABEL_SMOOTHING: 0.05 - 0.1 reduces overconfidence on imbalanced / noisy labels.
LABEL_SMOOTHING: float = 0.0


# == TRAINING ==
EPOCHS: int = 150
BATCH_SIZE: int = 32
LEARNING_RATE: float = 1e-4
# :param LR_SCHEDULER: 'cyclic' (default) cycles LR up to 20x the base LR; if a run diverges
#   (see the divergence axis), set this to None.
LR_SCHEDULER: Optional[str] = 'cyclic'
# Skip the slow HDBSCAN concept-clustering post-analysis unless you need it.
DO_CLUSTERING: bool = False
# Embedding-health diagnostics write many extra PNGs / .mp4 videos into the archive that this
# workflow never uses. Turn them off to keep runs faster and the log/output clean. Set any back to
# True only if you specifically want to inspect the embedding space.
TRACK_EMBEDDING_HEALTH: bool = False
TRACK_EMBEDDING_UNIFORMITY: bool = False
TRACK_EMBEDDING_TARGET_ALIGNMENT: bool = False
TRACK_EMBEDDING_PROJECTIONS: bool = False


# Keep __DEBUG__ = True so results land in a predictable .../<experiment>/debug/ folder that the
# skill's "read .../debug/report.json" steps rely on (False -> a timestamped archive folder instead).
__DEBUG__ = True
__TESTING__ = False

experiment = Experiment.extend(
    'train_model__megan.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.testing
def testing(e: Experiment):
    # Smoke-test configuration: a few epochs just to verify data loading + wiring.
    e.EPOCHS = 3
    e.NUM_EXAMPLES = 8


experiment.run_if_main()
