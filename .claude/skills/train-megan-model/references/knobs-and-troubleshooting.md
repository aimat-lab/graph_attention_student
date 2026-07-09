# MEGAN explanation knobs & troubleshooting

How to read a failed diagnostic self-check and decide which knob to change. All knob names are
the UPPERCASE parameters you set in your `train_model__megan__<dataset>.py` sub-experiment.

## The primary lever: `IMPORTANCE_OFFSET`

`IMPORTANCE_OFFSET` (default `0.8`) is by far the most important explanation knob. It controls how
much of each graph the explanations cover and interacts strongly with the other explanation
behaviours (accuracy, separation, sparsity). **Whenever the explanations are unsatisfactory —
coverage wrong, low `explanation_accuracy`, weak channel separation — the FIRST investigation is
to sweep `IMPORTANCE_OFFSET` in increments of 0.2** (e.g. 0.4, 0.6, 0.8, 1.0, 1.2, 1.4 …; useful
range ≈ 0.2–2.0), rerunning and reading the **validation** `report.json` at each step (keep the
sweep small, ≤ ~5 settings, and select on validation — never on the test report).
- **Higher** → more nodes highlighted (more coverage), less sparse.
- **Lower** → more focused / sparse masks.

Do this sweep before touching anything else. It resolves the majority of explanation problems.

## The secondary lever: `REGRESSION_MARGIN` (regression only)

`REGRESSION_MARGIN` (default `0.0`) sharpens the negative/positive class split used for explanation
co-training. It is measured **in units of the target's standard deviation**: a value `m > 0`
excludes samples within `mean ± m·std` of the (per-batch) target mean from the explanation loss,
so only *clearly* negative/positive molecules drive the masks → cleaner, better-separated
explanations. Keep it `0.0` by default; in harder cases raise it to `0.1` or `0.2`.
- Only **positive** values have an effect. Negative values currently behave identically to `0.0`
  in the code — do not use them.

## The core tension
**Sparsity and explanation-accuracy are antagonists.** Sweeping `IMPORTANCE_OFFSET` down (sparser)
too far drops the explanation-accuracy AUC toward random; too high dilutes the signal across the
whole graph. Aim for the middle of the sweep where AUC is clearly above chance AND the masks look
focused in `examples.png` — not a maximum of either alone. Change one knob at a time and re-check.

## About `REGRESSION_REFERENCE` — NOT a tuning knob
`REGRESSION_REFERENCE` is **auto-managed**: during regression training the model overwrites it with
a running mean of the true targets, and the co-training split uses the per-batch target mean — not
this parameter. It only affects output centering. **Leave it at the default (`0.0`); changing or
"flipping" it does nothing useful.** For channel-sign problems use `REGRESSION_MARGIN` / more
training / `FIDELITY_FACTOR` (below), not this.

## Failure → knob table

Read `report.json` → `overall.reasons`, then find the failing axis.

### `explanation_accuracy` FAIL/WARN — the explanation subgraph can't solve the proxy task
First check `computed`: if `computed: false`, this is a **measurement problem, not a bad model**
(e.g. a reloaded model missing `importance_mode`, or a single-class eval split) — fix that, don't
tune knobs.
If genuinely low (`per_channel_auc` near 0.5):
1. **Sweep `IMPORTANCE_OFFSET` in 0.2 steps** (primary). Over-sparse masks have nothing to separate
   on; over-broad masks dilute the signal — the sweep finds the balance.
2. **Raise `REGRESSION_MARGIN` to 0.1–0.2** (regression) for cleaner class separation.
3. **Train longer** (`EPOCHS`) — explanations form later than the prediction.
4. **Raise `IMPORTANCE_FACTOR`** (e.g. 1.0 → 2.0) to prioritize explanation co-training.

### `coverage` WARN — masks saturated (cover the whole molecule)
`frac_nodes_active` near 1.0 / `frac_graphs_saturated` high.
1. **Lower `IMPORTANCE_OFFSET`** (primary; step down in 0.2).
2. Try **`ATTENTION_AGGREGATION = 'min'`** — tends to produce the sparsest masks.
Watch `explanation_accuracy` while doing this (the antagonism).

### `coverage` WARN — masks empty/invisible
`frac_graphs_empty` high / `mean_importance` near 0.
1. **Raise `IMPORTANCE_OFFSET`** (primary; step up in 0.2).
2. **Raise `IMPORTANCE_FACTOR`** if explanations never formed at all.

### `fidelity_sign` FAIL — leaving a channel out moves the output the WRONG way
Uncommon for regression (the split is the per-batch mean with a canonical channel order). To fix:
1. **Raise `REGRESSION_MARGIN` to 0.1–0.2** so only clearly-signed samples train the channels.
2. **Set `FIDELITY_FACTOR` > 0** (e.g. 0.1) — directly rewards channels with the correct fidelity
   sign (via the fidelity loss). Use as a secondary enforcement.
3. If only mildly below threshold (WARN), more training often resolves it.

### `fidelity_magnitude` WARN — masking a channel barely changes the output
- **Raise `IMPORTANCE_FACTOR`** so the model routes more of its decision through the channels.
- **Set `FIDELITY_FACTOR` > 0** to reward channels that actually move the output.

### `prediction` WARN — the base predictor is weak (context)
Explanations are only meaningful if the prediction is decent.
- Increase capacity (`UNITS`, `FINAL_UNITS`), tune `LEARNING_RATE`, train longer (`EPOCHS`).
- **Classification with class imbalance** (a common cause): see the imbalance section below.
- Check the target scale/units and that the right column is being read.

### `divergence` FAIL — NaN output or exploded explanation loss
- **Lower `LEARNING_RATE`.**
- **Set `LR_SCHEDULER = None`** — the default `'cyclic'` scheduler peaks at ~20× the base LR and is
  a common divergence source.
- **Lower `IMPORTANCE_FACTOR`** so the explanation loss doesn't destabilize early training.

## Classification with class imbalance
A skewed dataset can collapse to majority-class predictions (poor `prediction` / `f1_macro`).
Detect it in step 1 by checking class counts. Remedies (parameters exist in the base experiment):
- **`CLASS_OVERSAMPLING = True`** — oversamples minority classes toward balance.
- **`OVERSAMPLING_FACTORS = {class_index: factor}`** — manual per-class oversampling control.
- **`LABEL_SMOOTHING` (e.g. 0.05–0.1)** — reduces overconfidence, helps generalization.
- **`OUTPUT_NORM`** — caps logit magnitude to further curb overconfidence.

## Knob quick-reference
| Knob | Effect | Typical range |
|------|--------|---------------|
| `IMPORTANCE_OFFSET` | **PRIMARY** coverage/explanation lever; sweep in 0.2 steps first | 0.2 – 2.0 |
| `REGRESSION_MARGIN` | (reg.) exclude near-mean samples (std units) for cleaner masks; positive only | 0.0 – 0.2 |
| `IMPORTANCE_FACTOR` | strength of explanation co-training | 0.5 – 3.0 |
| `ATTENTION_AGGREGATION` | layer attention aggregation; `'min'` = sparsest | max/min/sum/mean |
| `FIDELITY_FACTOR` | secondary: enforce correct fidelity sign / magnitude (0 = off) | 0.0 – 0.3 |
| `LR_SCHEDULER` | `None` to stop cyclic-LR divergence | 'cyclic' / None |
| `CLASS_OVERSAMPLING` | (clf) balance minority classes | True / False |
| `SPARSITY_FACTOR` | (marked DEPRECATED; still applies a Hoyer sparsity reg) extra sparsity | 0.0 – 3.0 |

`REGRESSION_REFERENCE` is intentionally absent — it is auto-managed, not a tuning knob (see above).

## Rule of thumb
1. Fix `divergence` and `prediction` first — nothing else matters if those fail.
2. For any explanation problem, **sweep `IMPORTANCE_OFFSET` in 0.2 increments first** — it is the
   primary lever and resolves most issues.
3. Then reach for `REGRESSION_MARGIN` (cleaner separation), then `IMPORTANCE_FACTOR`,
   `ATTENTION_AGGREGATION='min'`, `FIDELITY_FACTOR`.
4. Judge `coverage` visually via `examples.png`, watching that explanation-accuracy holds.
5. Change **one knob at a time**, rerun, re-read the report.
