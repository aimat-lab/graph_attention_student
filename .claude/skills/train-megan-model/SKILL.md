---
name: train-megan-model
description: >-
  Train a MEGAN self-explaining graph neural network on a new molecular dataset (a CSV of
  SMILES + target values) and judge whether the resulting explanations are usable. Use when
  asked to train MEGAN, build an explainable molecular property-prediction model, produce
  attribution/explanations for a molecular dataset, or "train a model on this CSV of molecules
  and report the explanations". Covers: writing a pycomex sub-experiment that extends
  train_model__megan.py, smoke-testing then running it, reading the automatic post-training
  diagnostic (report.json + scorecard/examples/fidelity images), tuning the explanation knobs
  when a run fails its self-check, and assembling a human-facing explanation report.
---

# Train a MEGAN model on a new molecular dataset

MEGAN is a self-explaining GNN: it predicts a molecular property AND produces per-node/edge
importance masks across multiple **explanation channels**. Getting *usable* explanations (not
just a good prediction) needs per-dataset tuning, and individual runs can converge to bad
explanations even when the prediction is fine. This skill drives that end-to-end and uses the
built-in **post-training diagnostic** as an objective self-check.

Always work inside the project venv: `source .venv/bin/activate`.

## The workflow

### 1. Understand and validate the data
- Confirm the CSV path, the SMILES column, the target column(s), and whether the task is
  **regression** (continuous target) or **classification** (discrete integer classes).
- **Sanity-check the CSV before training** — the loader *silently drops* rows with unparseable
  SMILES or blank/non-numeric targets, so validate up front (replace the placeholders):
  ```bash
  python - <<'PY'
  import pandas as pd
  df = pd.read_csv('YOUR.csv')
  print('rows:', len(df))
  t = 'TARGET_COL'
  print('null/blank targets:', df[t].isna().sum())
  print('non-numeric targets:', pd.to_numeric(df[t], errors='coerce').isna().sum())
  print('regression: target min/mean/max:', df[t].min(), df[t].mean(), df[t].max())
  print('classification: class counts:\n', df[t].value_counts())
  PY
  ```
- For **classification**, check the class balance here — a heavily skewed dataset needs the
  oversampling knobs (`CLASS_OVERSAMPLING` / `OVERSAMPLING_FACTORS`; see knobs-and-troubleshooting.md).
- After every run, read the loader's `loaded N, filtered F, skipped S` log line — it goes to the
  experiment log, not obvious stdout, so find it with
  `grep "loaded" .../results/train_model__megan__<dataset>/debug/experiment_out.log`. If `skipped` /
  `filtered` is a non-trivial fraction of the rows, investigate before trusting the model — you may
  be training on a decimated or biased subset.

### 2. Create the sub-experiment
- Copy `templates/train_model__megan__TEMPLATE.py` into **`graph_attention_student/experiments/`**
  and rename it `train_model__megan__<dataset>.py`.
  - **It MUST live in that directory** — pycomex resolves the base experiment by filename relative
    to the sub-experiment's own folder. A file elsewhere cannot `Experiment.extend('train_model__megan.py')`.
  - The filename (minus `.py`) becomes the experiment namespace and the archive folder name.
- Fill in every `# TODO`: CSV path, columns, `DATASET_TYPE`, `NUM_CHANNELS`, `FINAL_UNITS` last
  value, and `CHANNEL_INFOS` (what each channel *means* for this task). For classification also set
  `NUM_CLASSES`. Leave the explanation knobs at their defaults — tune them only in step 5.
- Start from the default knob values in the template. Do not pre-tune; tune only in response to
  a failed diagnostic (step 5).

### 3. Smoke test (always, before the full run)
- Set `__TESTING__ = True` in the file and run it: `python graph_attention_student/experiments/train_model__megan__<dataset>.py`
- The `@experiment.testing` hook drops it to ~3 epochs. This only checks that the CSV loads, the
  columns are right, SMILES parse, and nothing crashes — it will NOT produce good explanations.
- Fix any config/data errors here (cheap) before committing to a full run.

### 4. Full run
- Set `__TESTING__ = False` and run the same command. Training on GPU is strongly preferred.
  - **Environment note:** if training dies with `CUDA error: no kernel image is available`, the
    installed torch build lacks kernels for this GPU. Either install a matching CUDA build of
    torch, or fall back to CPU by prefixing the command with `CUDA_VISIBLE_DEVICES=""`.
- **What to expect:** a full run is `EPOCHS` epochs (a few seconds/epoch on GPU) followed by a
  heavier analysis/evaluation phase. It writes **many auxiliary files** into the archive — the
  diagnostic reports/images (below) plus supporting PDFs. Only `report.json` / `report_test.json`
  and the `scorecard`/`examples`/`fidelity` images matter; ignore the rest. A noisy log is normal.
- Output (including the diagnostic) is written to
  `graph_attention_student/experiments/results/train_model__megan__<dataset>/**debug**/`. The folder
  is literally named `debug` **because the template sets `__DEBUG__ = True`** — keep it that way so
  the `.../debug/report.json` paths in step 5 resolve. (With `__DEBUG__ = False` the archive becomes
  a timestamped folder you'd have to locate.)

### 5. Read the diagnostic self-check — the key step
The run writes **two** diagnostics into `.../debug/`:
- **`report.json`** (+ `scorecard.png` / `examples.png` / `fidelity.png`) — computed on the
  **validation** set. This is the **tuning gate**: read it to decide if a run is good and to
  compare knob settings across runs.
- **`report_test.json`** (+ `test_*.png`) — computed on the held-out **test** set. This is the
  **final, unbiased** verdict. Quote it once, for your chosen configuration — never tune against it.

Read `.../debug/report.json` and branch on `overall.verdict` (`PASS` / `WARN` / `FAIL`):
- First check `axes.explanation_accuracy.computed`: if `false`, the metric could not be measured (a
  bug, not a random model — see interpreting-the-report.md) → do NOT sweep knobs.
- **PASS** → explanations are usable on validation; confirm on `report_test.json`, then step 6.
- **WARN / FAIL** → read `overall.reasons` + per-axis values, then consult
  **`references/knobs-and-troubleshooting.md`** (see "Deciding whether to rerun" below).

Because coverage is a **visual** judgement, **open the images with the Read tool** (it renders
PNGs): `Read .../debug/scorecard.png` and `.../debug/examples.png` (use the `test_`-prefixed files
for the test set). If `examples.png` is absent, coverage was not visually verifiable — don't pass
the run on coverage grounds alone. Full field-by-field meaning is in
**`references/interpreting-the-report.md`**.

### 6. Deliver
- If the human asked for a report/explanations, assemble a human-facing summary following
  **`references/human-report.md`** (this synthesizes the run's artifacts into what the user
  actually asked for — separate from the fixed diagnostic).
- Always state the diagnostic verdict plainly (e.g. "explanations passed the self-check" or
  "explanations failed the fidelity-sign check — I'd recommend rerunning with X").

## Deciding whether to rerun (remediation philosophy)
The diagnostic **detects and flags**; it does not auto-tune. Keep model selection honest:
- **Tune against `report.json` (validation), never `report_test.json` (test)** — sweeping knobs and
  picking the best *test* report overfits the test set and inflates the final numbers.
- Runs are **deterministic for a fixed `SEED`**, so a change in `report.json` between runs reflects
  the knob you changed, not RNG noise. (For a robustness estimate, rerun with a different `SEED`.)
- On WARN/FAIL: identify the failing axis → knob (`references/knobs-and-troubleshooting.md`). For any
  explanation-quality failure, first **sweep `IMPORTANCE_OFFSET` in 0.2 increments** — keep the sweep
  small (≤ ~5 settings) so you don't overfit even the validation set.
- Each rerun **overwrites `debug/`**; copy `report.json` (and `scorecard.png`) aside between attempts
  to compare, and keep a small table of `offset → val verdict / mean_auc`.
- Once you pick the best config, do **one** final run and **report `report_test.json`** as the headline.
- If autonomy was implied ("train it and give me a good model"), run this loop within a small budget
  (e.g. 2–3 attempts) then report; if hands-on, report the diagnosis + recommended knob and ask first.
- A run whose **prediction** is good but **explanations** FAIL is common and is exactly what this
  loop exists to catch — do not ship it as "done" without saying the explanations did not pass.

## Key facts to keep straight
- **Regression** → `NUM_CHANNELS = 2` (channel 0 = negative, 1 = positive); `FINAL_UNITS[-1]` = #targets.
- **Classification** → `NUM_CHANNELS` = #classes; `FINAL_UNITS[-1]` = #classes; set `NUM_CLASSES`.
- **`IMPORTANCE_OFFSET` is the primary explanation lever** — sweep it in 0.2 increments as the
  first response to any explanation problem (coverage, low accuracy, weak separation).
  `REGRESSION_MARGIN` (default 0, raise to 0.1–0.2 in std units) is the secondary lever.
- `REGRESSION_REFERENCE` is **auto-managed** (the model overwrites it with the running target mean
  during training) and is NOT a tuning knob — leave it at the default.
- Sparsity and explanation-accuracy are **antagonists** — pushing masks sparser can drop the
  explanation-accuracy AUC. Aim for "in the right ballpark", not a maximum of either alone.
