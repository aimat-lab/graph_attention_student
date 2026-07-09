# Interpreting the diagnostic report

After a run, the post-training diagnostic writes **two** report sets into the archive folder
`graph_attention_student/experiments/results/train_model__megan__<dataset>/debug/`:

- **Validation gate** — `report.json` + `scorecard.png`, `examples.png`, `fidelity.png`
  (computed on the **validation** set). Use this to judge a run and to compare knob settings.
- **Test report** — `report_test.json` + `test_scorecard.png`, `test_examples.png`,
  `test_fidelity.png` (computed on the held-out **test** set). The final unbiased estimate; quote it
  once for the chosen config, never tune against it.

Both have identical structure (below). Reads are byte-reproducible for a fixed `SEED`.

`report.json` is self-sufficient — every verdict ships with the raw numbers behind it, so you can
gate on the JSON alone. But **coverage is a visual judgement**: **open the images with the Read tool**
(it renders PNGs) — `Read .../debug/scorecard.png` and `.../debug/examples.png` — before concluding
the masks are good. If `examples.png` is missing (rendering needs node positions + background
images), fall back to the coverage bars in `scorecard.png` and don't pass on coverage alone.

## `report.json` structure

```jsonc
{
  "overall": { "verdict": "PASS|WARN|FAIL", "reasons": [...], "hard_gates": {...} },
  "meta":    { "prediction_mode", "num_channels", "n_eval_graphs", "regression_reference" },
  "divergence": { "verdict", "nan_in_output", "expl_loss_exploded", "final_expl_loss" },
  "axes": { "prediction", "explanation_accuracy", "fidelity_sign",
            "fidelity_magnitude", "coverage" }
}
```

- **`overall.verdict`** — read this first. `FAIL` if any hard gate fails (`divergence`,
  `explanation_accuracy`, `fidelity_sign`), else the worst axis verdict.
- **`overall.reasons`** — one line per non-PASS axis. Your to-do list.

## The five axes

### `prediction` (context; never hard-fails)
- Regression: `r2`, `mae`, `rmse`. Classification: `accuracy`, `f1_macro`.
- WARN only if the predictor is weak. A weak predictor makes the explanations moot.

### `explanation_accuracy` — HARD FAIL GATE
- `per_channel_auc`, `mean_auc`: AUC of the explanation proxy task (can the mask subgraph alone
  separate the classes?). ~0.5 = no signal; ≥0.7 good.
- **`computed` (true/false)**: if `false`, the metric could NOT be measured (with an `error`
  string) — e.g. the model lacks `importance_mode`, or the eval set was single-class. This is a
  **broken measurement, not a random model**: fix the cause, don't tune knobs.
- `uncomputable_channels` (optional): channels that couldn't be scored; `mean_auc` is over the rest.

### `fidelity_sign`
- `per_channel_sign_consistency`: fraction of graphs whose leave-one-out deviation sits on the
  canonically-expected side (≥0.5 majority-correct; ≥0.7 good).
- `per_channel_mean_deviation`: signed mean deviation. For regression, channel 0 should be
  **negative**, channel 1 **positive**. Inverted signs → see knobs-and-troubleshooting.md (raise
  `REGRESSION_MARGIN`, set `FIDELITY_FACTOR` > 0, or train longer). `REGRESSION_REFERENCE` is
  auto-managed and is **not** the fix.

### `fidelity_magnitude`
- `per_channel_relative_to_target_scale`: how much masking a channel moves the output, relative
  to the target's spread. WARN if all channels are negligibly small.

### `coverage` (soft; judged visually)
Per channel: `mean_importance`, `frac_nodes_active`, `frac_graphs_empty`, `frac_graphs_saturated`.
- Good ≈ masks focused on a meaningful fraction of the molecule, not ~0 (invisible) and not ~1
  (whole molecule). The verdict only flags the extremes — **the real check is `examples.png`.**

## The images
- **`scorecard.png`** — one-glance summary: per-channel explanation-AUC vs coverage (the
  antagonism), plus every verdict and the overall verdict. Best single image to view.
- **`examples.png`** — random example explanations, rows = channels. Confirm masks are focused and
  land on sensible substructures (e.g. for solubility: polar groups on the positive channel,
  hydrophobic groups on the negative channel).
- **`fidelity.png`** — leave-one-out deviation histograms per (channel × target); the sign check
  in distribution form.

## Quick verdict → action
| overall | meaning | action |
|---------|---------|--------|
| PASS | explanations usable | deliver; glance at examples.png to confirm coverage |
| WARN | usable but imperfect | note it; optionally tune (knobs-and-troubleshooting.md) |
| FAIL | not usable as-is | read reasons → knob → decide whether to rerun |

If `explanation_accuracy.computed == false`, treat it as a **bug to fix**, not a tuning problem.
