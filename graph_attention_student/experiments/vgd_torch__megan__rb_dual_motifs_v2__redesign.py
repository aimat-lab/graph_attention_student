"""
MEGAN on rb_dual_motifs_v2 with the redesigned explanation loss.

This is the configuration arrived at after a long investigation into why the model omitted the
green vertex of ring motifs entirely - its attention sat at 0.0086, BELOW the 0.056 background -
while covering star motifs almost perfectly.

                                baseline      this configuration
    ring vertex in mask          0.0086            0.991
    ring leaf                    0.637             0.994
    star hub / leaf          0.966 / 0.895     0.996 / 0.996
    decoy attention vs bg       7x / 13x       0.38x / 0.44x
    test R2                      0.947             0.96

Reproduced across two random seeds.

THE CAUSE. A ring can be told from a near-miss either by the edge that closes it or by its two
spokes, and both are valid discriminators. But node importance is pooled from a node's INCIDENT
edges, and the closing edge is not incident to the green vertex - so only the spoke route covers
it. Sparsity at full strength from the first step saturated the spokes before any completeness term
could defend them, and a sigmoid driven into its tail has no gradient left to recover with, which
forced the negative channel onto the closing-edge solution permanently.

WHAT FIXES IT, and each part was verified to do what it claims rather than assumed:

  EXPLANATION_OBJECTIVE = difference_margin
      The original objective drives each channel's pooled importance to a fixed setpoint. Being a
      sum, it is invariant to spreading the same mass over more nodes, so it says how much to
      highlight and never where; and the setpoint was never reached, making it permanent upward
      pressure. This asks only that the on-channel exceed the off-channel by a margin scaled to the
      target's distance from the mean - a one-sided condition that can be satisfied, after which it
      stops pulling. Raised precision from 0.48 to 0.92 on its own.

  RDT_FACTOR with RDT_WARMUP = RDT_RAMP = 0
      Randomise everything outside the mask and require the prediction to survive. This is what
      makes a node's membership NECESSARY rather than merely consistent - leave the green vertex
      out and it gets randomised, the ring stops being a ring, and the prediction moves. Active
      from the first step is essential: ramping it in let the spokes die before it could defend
      them, which is the whole failure above.

  RDT_MASK_ONLY
      The term backpropagates through a second forward pass, so without this its gradient reaches
      every weight and the cheaper way to satisfy it is to make the network robust to noise rather
      than to fix the mask. Measured: 168 parameters received its gradient, 118 with this enabled.

  RDT_USE_RAW
      Blend on the RAW importance, not the normalized mask. Otherwise the model can protect more
      nodes by lowering the per-graph maximum instead of raising the others, which collapses the
      reported scale - measured, the maximum fell to 0.507 against 0.902 - and leaves a mask that
      is structurally right but numerically unusable at a 0.5 threshold.

  POLAR_FACTOR
      Penalises x(1-x) so values commit to 0 or 1. Nothing else constrains the raw scale, because
      every other term divides by the per-graph maximum; without this the absolute values drift and
      a run can reach recall 0.99 at threshold 0.1 while reading 0.083 at 0.5.

  SPREAD_FACTOR = 0.5
      Charges for the effective support size of the mask. Best of the sweep: ring vertex 0.991 and
      the lowest excess node count of any arm, 0.23 against 0.52 at 0.1.

REGRESSION_MARGIN is positive so the margin block actually runs - the implementation guards it with
'if regression_margin > 0', so the negative default silently disabled it.
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
#       Calibrate per dataset. This constant does NOT transfer - the same value was satisfied on 6%
#       of graphs here before recalibration, 83% on aqsoldb and 55% on mutagenicity. The logged
#       'margin_met' metric reports the satisfied fraction each epoch; aim for roughly half.
DIFFERENCE_MARGIN_SCALE: float = 4.0
# :param REGRESSION_MARGIN:
#       Must be positive to take effect at all.
REGRESSION_MARGIN: t.Optional[float] = 0.1

# == RDT FIDELITY ==

# :param RDT_FACTOR:
RDT_FACTOR: float = 1.0
# :param RDT_MASK_ONLY:
RDT_MASK_ONLY: bool = True
# :param RDT_USE_RAW:
RDT_USE_RAW: bool = True
# :param RDT_WARMUP:
#       Zero. Ramping this in is what let the spokes saturate before the term could defend them.
RDT_WARMUP: float = 0.0
# :param RDT_RAMP:
RDT_RAMP: float = 0.0
# :param RDT_SAMPLES:
#       One. Four were tried and tracked the single-sample loss closely, so the extra cost bought
#       nothing here.
RDT_SAMPLES: int = 1

# == MASK SHAPE ==

# :param POLAR_FACTOR:
POLAR_FACTOR: float = 0.3
# :param SPREAD_FACTOR:
SPREAD_FACTOR: float = 0.5
# :param SPARSITY_FACTOR:
#       Held at 1.0. Three runs at 0.3 put both channels on the closing-edge solution and lost the
#       ring vertex entirely.
SPARSITY_FACTOR: float = 1.0


__DEBUG__ = True
__TESTING__ = False

experiment = Experiment.extend(
    'vgd_torch__megan__rb_dual_motifs_v2.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()
