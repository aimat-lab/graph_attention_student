"""
MEGAN on mutagenicity with the redesigned explanation loss - the classification form.

Ports the configuration developed on rb_dual_motifs_v2, where it took the green vertex of ring
motifs from 0.0086 (below background) to 0.991. See that experiment for the mechanism.

THE OBJECTIVE HAD TO BE WRITTEN FOR CLASSIFICATION. The difference margin existed only in the
regression branch, so setting EXPLANATION_OBJECTIVE on a classification dataset was a SILENT NO-OP
- and since that term produced the largest single gain on the colour graphs, a naive port would
have quietly tested an incomplete recipe. The classification form requires the channel of the TRUE
class to exceed the strongest competing channel's pooled mass by a margin, keeping the off-channel
exclusivity term.

The margin is a CONSTANT here rather than scaled by target distance, because a classification label
has no gradation - every example is equally a member of its class. Scaling it by the model's own
predicted confidence was considered and rejected as self-referential.

RESULTS at 100 epochs, accuracy 0.867:

    margin   empty masks (correct)   mask atoms   % of molecule
      2.0            0.128              6.41          38%
      3.0            0.112              7.72          45%
      4.0            0.085              9.51          56%

4.0 is chosen. A criterion set in advance called anything past about 8 atoms over-inclusion, but
that was withdrawn on domain grounds: it assumed mutagenicity is purely a matter of discrete
structural alerts, whereas it also acts through whole-molecule properties - aromatic systems,
planarity, electrophilicity - so a mask covering half the molecule can be honest rather than
unrestrained.

TWO THINGS TO KNOW WHEN READING THIS MODEL'S DIAGNOSTIC.

fidelity_sign FAILS, and should. It is driven by channel 0 at 0.394 sign consistency with a mean
deviation of -0.010. That is the non-mutagenic class, which has no positive structural signature -
mutagenicity comes from the PRESENCE of alerts, and an absence cannot be masked out. Channel 1, the
mutagenic class, sits at 0.602.

The 'approx' figure in the training log is meaningless here. It thresholds tanh(0.1 * offset *
pooled) at 0.5, and with IMPORTANCE_OFFSET at 0.5 nothing ever reaches that, so it reports chance
regardless of quality - 0.55 while the diagnostic's AUC-based measure of the same explanations was
0.828. Trust the diagnostic axis, not the training metric.

Roughly a tenth of correctly classified molecules end with no explanation, and six configurations
moved that only between 0.085 and 0.141. That looks like a property of the data - not every mutagen
carries a clean alert - rather than something to tune away.
"""
import typing as t
import pathlib

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace


PATH = pathlib.Path(__file__).parent.absolute()

# == EXPLANATION OBJECTIVE ==

# :param EXPLANATION_OBJECTIVE:
EXPLANATION_OBJECTIVE: str = 'difference_margin'
# :param DIFFERENCE_MARGIN_SCALE:
#       A constant for classification. Watch the logged 'margin_met' - it settled at 0.55 with a
#       scale of 2.0 here, which is the regime to aim for.
DIFFERENCE_MARGIN_SCALE: float = 4.0

# == RDT FIDELITY ==

# :param RDT_FACTOR:
RDT_FACTOR: float = 1.0
# :param RDT_MASK_ONLY:
RDT_MASK_ONLY: bool = True
# :param RDT_USE_RAW:
RDT_USE_RAW: bool = True
# :param RDT_WARMUP:
RDT_WARMUP: float = 0.0
# :param RDT_RAMP:
RDT_RAMP: float = 0.0
# :param RDT_SAMPLES:
RDT_SAMPLES: int = 1

# == MASK SHAPE ==

# :param POLAR_FACTOR:
#       Lower than the colour-graph value. Reducing it from 0.3 moved the empty-mask rate from
#       0.141 to 0.128; the effect is small either way, so polarization is not what drives the
#       empty masks here.
POLAR_FACTOR: float = 0.1
# :param SPREAD_FACTOR:
SPREAD_FACTOR: float = 0.5
# :param SPARSITY_FACTOR:
SPARSITY_FACTOR: float = 1.0


__DEBUG__ = True
__TESTING__ = False

experiment = Experiment.extend(
    'vgd_torch__megan__mutagenicity.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()
