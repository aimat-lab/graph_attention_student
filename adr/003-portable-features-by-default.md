# ADR-003: Train on the Portable Feature Set by Default

**Date:** 2026-10-20
**Status:** Accepted

## Context

ADR-002 made trained models exportable to ONNX, but a browser can only run a model whose
input features it can compute. The CSV training experiment, `train_model__megan.py`, used
the full molecule processing: 45 node features, five of which are per-atom RDKit
descriptors (Crippen, TPSA, Labute ASA, Gasteiger, EState) that RDKit.js cannot compute.
Its models therefore exported without a featurization contract, and the standard training
workflow never produced a model that could be deployed to a browser.

The cost of dropping those five descriptors was measured on AqSolDB, over five paired
seeds that differed only in the node features (`vgd_torch__megan__aqsoldb__featureset*.py`).
The resulting 38-feature portable set was statistically indistinguishable from the full set
on test R² (−0.009, p = 0.18) and lost a little explanation AUC (−0.024, p = 0.03).

## Decision

`train_model__megan.py` gains two parameters, both on by default:

- `PORTABLE_FEATURES` trains on `PortableMoleculeProcessing`. It replaces only the default
  `PROCESSING_PATH`, so a processing module the user chose still takes precedence. It
  points `PROCESSING_PATH` at `portable_processing.py` rather than returning a different
  processing object, because the base experiment archives the module at that path as
  `process.py`. The archive therefore always holds the processing the model was trained
  with — which the feature-set runs did not, having archived the dataset's 45-column module
  next to a 38-column model.
- `EXPORT_ONNX` exports the saved model into `<archive>/onnx_export`, the same layout
  `megan export` writes. A failed export, including a missing `export` extra, is logged as
  a warning and never fails the run: the model is already saved by then, and losing a
  finished training run to an optional artifact would be the worse outcome.

## Consequences

**Good**

- With the `export` extra installed, every run of the standard workflow is deployable to a
  browser without retraining or re-exporting.
- The archived `process.py` matches the model, so `megan export` works inside any archive
  with its defaults.

**Bad / accepted costs**

- On AqSolDB, explanation AUC is about 0.02 lower than with the full feature set. That is
  one dataset; the cost elsewhere is unmeasured.
- Existing sub-experiments that set neither flag change behaviour: they now train on 38
  instead of 45 node features and produce slightly different models.
  `PORTABLE_FEATURES = False` restores the previous behaviour.
- Without the `export` extra, every run ends with a warning about the skipped export.

## Alternatives considered

**An opt-in flag.** Keeps existing configurations reproducing exactly, but leaves the
standard workflow producing models that cannot reach a browser unless the user knows to ask.

**Documenting `PROCESSING_PATH` only.** Needs no code, but requires every user to know
which processing module is portable, and leaves the default workflow unable to reach a
browser.
