# Exporting MEGAN Models to ONNX

A trained MEGAN model can be exported to [ONNX](https://onnx.ai), which runs it outside
this Python stack: in a Python service without PyTorch, in Node, or directly in the
browser. The browser case needs no inference server at all — a static page can predict a
property and draw the explanation with nothing but files.

The ONNX graph takes graph tensors, not SMILES, because RDKit's sanitization and
aromaticity perception cannot be expressed as ONNX operators. Every export therefore comes
with a machine-readable description of the featurization, and with `megan.js`, one
JavaScript file that implements it on top of
[RDKit.js](https://www.npmjs.com/package/@rdkit/rdkit) and predicts molecules from their
SMILES.

The export is verified under ONNX Runtime on the CPU in Python and with ONNX Runtime
Web's WASM backend in Node, both by the test suite, and in Chrome, by hand. The graph
avoids `ScatterElements`, which ONNX Runtime Web's WebGPU backend lacks, so it should run
there too, but that has not been tested. The design is recorded in
[ADR-002](adr/002-onnx-export.md), why training produces portable models by default in
[ADR-003](adr/003-portable-features-by-default.md), and why `megan.js` ships in every export
rather than as an npm package in [ADR-004](adr/004-javascript-runtime-in-every-export.md).

## What an export contains

Every export is a folder, by default named `onnx_export`, with these files:

| File | Contents |
|---|---|
| `model.onnx` | The model, self-contained (no external-data sidecar), typically 1–5 MB depending on layer sizes. Outputs predictions, node and edge importances, graph embeddings and the variance head. |
| `manifest.json` | The data contract: file hash, opset, input and output signature, the model architecture and configuration, and notes on behaviour that is easy to get wrong. |
| `featurization.json` | How to turn a molecule into the input tensors, column by column. Only written when every feature can be computed in JavaScript — see [The portable feature set](#the-portable-feature-set). |
| `megan.js` | The JavaScript runtime: the featurizer, the input batching and `MeganModel`, which predicts from SMILES. It is the copy from the package version that wrote the export, so it always matches `featurization.json`, and it is only written together with it. |
| `fixtures.json` | Example inputs with the outputs PyTorch produced for them, and the per-molecule features Python computed, so that a consumer can check itself. |

## Installing

The export needs the `export` extra, which provides `onnx`, `onnxruntime` and `onnxscript`:

```bash
uv pip install -e ".[export]"        # from a clone
pip install "graph_attention_student[export]"
```

Consumers need an ONNX runtime and something to featurize molecules: in Python,
`onnxruntime` and the model's `process.py` (which needs `visual_graph_datasets`); in
JavaScript, `onnxruntime-web`, `@rdkit/rdkit` and the export's own `megan.js`.

## Getting an exported model

### Automatically, while training

`train_model__megan.py`, the experiment that trains MEGAN from a CSV file, produces a
browser-ready model by default. It trains on the portable feature set and, once the model
is saved, exports it into the `onnx_export` folder of the run's archive (with the
default debug mode, `graph_attention_student/experiments/results/train_model__megan/debug/`):

```bash
python graph_attention_student/experiments/train_model__megan.py \
    --CSV_FILE_PATH='"/path/to/your/data.csv"' \
    --TARGET_COLUMN_NAMES='["target"]'
```

Two parameters control this:

| Parameter | Default | Effect |
|---|---|---|
| `PORTABLE_FEATURES` | `True` | Train on the 38 node / 11 edge features that `megan.js` can reproduce. `False` trains on the full 45 node features, for a model that is only ever used from Python. A `PROCESSING_PATH` of your own takes precedence over this flag. |
| `EXPORT_ONNX` | `True` | Export the saved model into `<archive>/onnx_export`. Without the `export` extra, or if the export fails, the run still completes; it logs a warning and records the reason in the experiment data under `onnx/error`. |

Why this is the default, and what it costs, is recorded in
[ADR-003](adr/003-portable-features-by-default.md). The archive's `process.py` is the
processing the model was trained with, so the archive folder can also be exported again by
hand, as below.

### Manually, from a checkpoint

Any trained `Megan` model can be exported, as long as you have the processing it was
trained with. From the command line, inside a folder with `model.ckpt` and `process.py` —
which is what `megan train` and the training experiments leave behind:

```bash
megan export
```

This writes `onnx_export/`. The options:

| Option | Default | |
|---|---|---|
| `--model-path` | `model.ckpt` | The checkpoint to export. |
| `--processing-path` | `process.py` | The processing module the model was trained with. |
| `--output-path` | `onnx_export` | The folder to write. |
| `--smiles` | four example molecules | Comma-separated SMILES used to trace the model and written to `fixtures.json`. Use several molecules of different sizes; molecules without bonds are not supported. |
| `--opset` | `18` | ONNX opset, at least 18. |

From Python:

```python
from graph_attention_student import Megan
from visual_graph_datasets.util import dynamic_import

model = Megan.load('model.ckpt')
processing = dynamic_import('process.py').processing

graphs = [processing.process(smiles) for smiles in ['CCO', 'c1ccccc1O', 'CC(=O)Oc1ccccc1C(=O)O']]
paths = model.export_onnx('onnx_export', graphs, processing=processing)
```

`paths` maps each written file's role (`model`, `manifest`, `featurization`, `javascript`,
`fixtures`) to its path. `export_megan_onnx` in `graph_attention_student.torch.export` is the same
function, for callers without a model method at hand.

The processing must be the one the model was trained with. The exporter checks the feature
widths before tracing and refuses a mismatch with a message naming both sizes.

**Training a portable model in your own code.** Use the portable processing wherever you
would use `MoleculeProcessing`, and keep its module as the run's `process.py`:

```python
import shutil
from graph_attention_student.torch.export import portable_processing

processing = portable_processing.processing  # a PortableMoleculeProcessing
shutil.copy(portable_processing.__file__, 'process.py')
```

`megan train` uses the default `MoleculeProcessing` (10 node / 4 edge features), which is
portable as well, so its models export with a `featurization.json` too.

**Models trained on the full feature set** export normally, but without
`featurization.json`; `manifest.json` says why under `featurizationOmittedBecause`. Such a
model can still run under ONNX Runtime from Python, featurized by its original
`process.py`, but a browser cannot compute its input features. Retrain it with the
portable feature set for that.

## The portable feature set

RDKit.js exposes no per-atom API. Five per-atom descriptors in the full featurization —
Crippen contributions, TPSA, Labute ASA, Gasteiger charges and EState indices — are
decompositions of molecule-level quantities that a browser cannot compute.
`PortableMoleculeProcessing` (`graph_attention_student/torch/export/portable_processing.py`)
is the full processing without them:

- **38 node features**: element (16 elements plus unknown), hybridization, total degree,
  number of hydrogens, mass, formal charge, aromaticity, ring membership.
- **11 edge features**: bond type, stereo, aromaticity, ring membership, conjugation.

Its columns are the first 38 of the full set, in the same order. Hybridization and
conjugation are not exposed by RDKit.js either; `megan.js` ports RDKit's own perception code
for them.

What dropping the descriptors costs was measured on AqSolDB, over five paired seeds (same
splits, model and explanation objective in every arm):

| Feature set | Node features | Test R² | Explanation AUC |
|---|---|---|---|
| full | 45 | 0.8552 ± 0.0130 | 0.9105 ± 0.0060 |
| portable | 38 | 0.8462 ± 0.0134 | 0.8869 ± 0.0146 |
| portable, also without hybridization | 32 | 0.8193 ± 0.0250 | 0.8863 ± 0.0087 |

The portable set is statistically indistinguishable from the full one on R² (−0.009,
p = 0.18) and costs a little explanation AUC (−0.024, p = 0.03). Dropping hybridization as
well costs four times as much R² (−0.036, p = 0.035), which is why it was ported rather
than dropped. The experiments are `vgd_torch__megan__aqsoldb__featureset*.py`, and
`scripts/analyze_featureset_results.py` reproduces the table from their results.

## Running an exported model

### Inputs and outputs

`K` is the number of explanation channels (`manifest.architecture.config.num_channels`),
`O` the number of outputs and `D` the embedding size. The node, edge and graph dimensions
are dynamic, so one session handles any batch.

| Input | Type | Shape | |
|---|---|---|---|
| `x` | float32 | (num_nodes, node_dim) | Node features. |
| `edge_index` | int64 | (2, num_edges) | All edge sources, then all edge targets. |
| `edge_attr` | float32 | (num_edges, edge_dim) | Edge features. |
| `batch` | int64 | (num_nodes,) | Which molecule each node belongs to. |
| `graph_ids` | int64 | (num_graphs,) | `0, 1, …, num_graphs - 1`. Only its length is read, as the batch size. |
| `node_mask` | float32 | (num_nodes, K) | All ones for a normal prediction. See [Explanation fidelity](#explanation-fidelity). |

| Output | Shape | |
|---|---|---|
| `graph_output` | (num_graphs, O) | The prediction. Raw values for regression, **logits** for classification. |
| `graph_embedding` | (num_graphs, D, K) | Per-channel graph embeddings. |
| `node_importance` | (num_nodes, K) | Raw node importances. |
| `node_importance_norm` | (num_nodes, K) | Node importances normalized per molecule (maximum 1), with values below 0.1 scaled down by 0.01. This is what an explanation drawing should use. |
| `edge_importance` | (num_edges, K) | Raw edge importances. |
| `edge_importance_norm` | (num_edges, K) | Edge importances, normalized like the node ones. |
| `graph_variance` | (num_graphs, 1) | The variance head; meaningful only for models trained with mean-variance estimation. |

For a regression model, channel 0 marks what lowers the prediction and channel 1 what
raises it. For classification, channel `k` marks evidence for class `k`.

### In Python, without PyTorch

The run's `process.py` featurizes; `onnxruntime` runs the model. Neither PyTorch nor
PyTorch Geometric is imported.

```python
import json

import numpy as np
import onnxruntime as ort
from visual_graph_datasets.util import dynamic_import

processing = dynamic_import('process.py').processing
manifest = json.load(open('onnx_export/manifest.json'))
session = ort.InferenceSession('onnx_export/model.onnx', providers=['CPUExecutionProvider'])

graph = processing.process('CC(=O)Oc1ccccc1C(=O)O')
num_nodes = len(graph['node_attributes'])
num_channels = manifest['architecture']['config']['num_channels']
outputs = session.run(None, {
    'x': graph['node_attributes'].astype(np.float32),
    'edge_index': graph['edge_indices'].T.astype(np.int64),
    'edge_attr': graph['edge_attributes'].astype(np.float32),
    'batch': np.zeros(num_nodes, dtype=np.int64),
    'graph_ids': np.zeros(1, dtype=np.int64),
    'node_mask': np.ones((num_nodes, num_channels), dtype=np.float32),
})
results = dict(zip([output['name'] for output in manifest['model']['outputs']], outputs))

print(results['graph_output'][0])       # the prediction
print(results['node_importance_norm'])  # (atoms, channels), 0..1
```

### In the browser

`megan.js` is one ES module with no dependencies of its own; you pass it RDKit.js and
ONNX Runtime Web. Both load from a CDN, so a static page needs nothing next to it but the
export folder. This page runs the model on aspirin and draws the molecule with the atoms
of channel 1 highlighted:

```html
<!doctype html>
<meta charset="utf-8">
<p id="prediction"></p>
<div id="drawing"></div>

<script src="https://cdn.jsdelivr.net/npm/@rdkit/rdkit@2025.3.2-1.0.0/dist/RDKit_minimal.js"></script>
<script type="module">
  import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.min.mjs';
  import { MeganModel } from './onnx_export/megan.js';

  const RDKit = await window.initRDKitModule();
  const model = await MeganModel.load('onnx_export/', { RDKit, ort });

  const smiles = 'CC(=O)Oc1ccccc1C(=O)O';
  const [result] = await model.predict([smiles]);
  document.getElementById('prediction').textContent = result.prediction[0];

  // Highlight the atoms that channel 1 marks, i.e. those at or above the 0.1 threshold.
  const atoms = result.nodeImportance.flatMap((channels, atom) => (channels[1] >= 0.1 ? [atom] : []));
  const mol = RDKit.get_mol(smiles);
  document.getElementById('drawing').innerHTML =
    mol.get_svg_with_highlights(JSON.stringify({ atoms }));
  mol.delete();
</script>
```

The versions in the two URLs are the ones the tests run against. To do without a CDN, copy
`RDKit_minimal.js` and `RDKit_minimal.wasm` from `@rdkit/rdkit/dist`, and `ort.min.mjs`
with the `ort-wasm-*.mjs` and `ort-wasm-*.wasm` files from `onnxruntime-web/dist`, next to
the page, and point the two URLs at the copies.

The node order is RDKit's atom order for the same SMILES, so `RDKit.get_mol(smiles)`
numbers the atoms exactly as the importances do. The page must be served over HTTP; it
cannot be opened as a `file://` URL. Multi-threading in ONNX Runtime Web needs a
cross-origin-isolated page; without it, as on GitHub Pages, it runs single-threaded, which
is plenty for one molecule at a time. The whole path takes about 4.5 ms per molecule,
featurization included, for the `units=[128, 128, 128]` AqSolDB model (Node, one thread).

**megan.js from a CDN.** jsDelivr serves the file straight from each release tag, for a
page that runs several models with one copy, or an online editor with no folder to put it
in:

```js
import { MeganModel } from 'https://cdn.jsdelivr.net/gh/aimat-lab/graph_attention_student@v1.7.0/graph_attention_student/torch/export/megan.js';
```

Use the version in the export's `manifest.json` (`generatedBy`). A `megan.js` that cannot
read an export's contract refuses it with an error naming the version that wrote it. Do not
link raw.githubusercontent.com instead: it serves files as `text/plain`, which browsers
refuse to run as a module.

### What `predict` returns

`model.predict(smilesList)` featurizes a list of SMILES, runs them through the session as
one batch and returns one result per molecule, in order:

| Field | |
|---|---|
| `smiles` | The input. |
| `prediction` | One value per target. Regression values as they are; for a classifier, probabilities (see [Classification models](#classification-models)). |
| `nodeImportance` | One row per atom, in RDKit's atom order, with one value per explanation channel. This is `node_importance_norm`, the output an explanation drawing should use. |
| `edgeImportance` | One row per edge, from `edge_importance_norm`. Every bond is two edges, forward then reverse, as listed in `graph.edgeIndices`. |
| `outputs` | Every model output for this molecule, by its name in the table above: one row for `graph_*` outputs, one row per atom or edge for `node_*` and `edge_*`. |
| `graph` | The featurized molecule: `nodeAttributes`, `edgeIndices` and `edgeAttributes`. |

A SMILES that RDKit cannot parse, or a molecule without bonds, fails the whole call with an
error naming it.

`predict` combines two steps that `megan.js` exports as well, for code that runs the
session itself: `model.featurizer.process(smiles)` (a `Featurizer`) turns a SMILES into
graph arrays, and `Featurizer.toFeeds(graphs, ort, numChannels)` batches graphs into the
input tensors exactly as PyTorch Geometric does.

### In Node

Install `onnxruntime-web` and `@rdkit/rdkit`, and import `megan.js` from the export folder.
`MeganModel.load` fetches over HTTP, and Node's `fetch` cannot read local files, so in Node
the model is put together from the files with the constructor:

```js
import fs from 'node:fs';
import { createRequire } from 'node:module';
import * as ort from 'onnxruntime-web';
import { MeganModel } from './onnx_export/megan.js';

const require = createRequire(import.meta.url);
const model = new MeganModel({
  manifest: JSON.parse(fs.readFileSync('onnx_export/manifest.json', 'utf8')),
  session: await ort.InferenceSession.create(fs.readFileSync('onnx_export/model.onnx')),
  RDKit: await require('@rdkit/rdkit')(),
  ort,
});

const [aspirin] = await model.predict(['CC(=O)Oc1ccccc1C(=O)O']);
console.log(aspirin.prediction);         // the prediction, one value per target
console.log(aspirin.nodeImportance[0]);  // atom 0, one value per channel
```

Node 22 and later load `megan.js` as an ES module wherever it sits. On older versions, add
`"type": "module"` to your project's `package.json`. `megan.js` only uses the runtime's
`Tensor` class and `InferenceSession`, so the code should be the same with
`onnxruntime-node`, which has not been tested.

### Several molecules at once

`predict` takes a list and runs it as one batch. Underneath, `toFeeds` batches graphs the
way PyTorch Geometric does, which is also how to batch the Python inputs: node and edge
arrays concatenated, edge indices offset by the number of atoms before them, and `batch`
holding each node's molecule index. The outputs come back concatenated in the same order —
`graph_*` has one row per molecule, `node_*` one block of rows per molecule (as many as it
has atoms), and `edge_*` one block per molecule with two rows per bond, forward then
reverse. `predict` splits them back into molecules; a caller of `toFeeds` has to do that
itself.

### Classification models

`graph_output` is raw logits; MEGAN applies no final activation. `predict` applies a
softmax, or a sigmoid per target for a `prediction_mode='bce'` model, and returns the
probabilities as `prediction`; the logits stay in `outputs.graph_output`. When you run the
session yourself, apply the activation yourself.

### Explanation fidelity

How much each channel actually moves the prediction — what
`Megan.leave_one_out_deviations` computes — needs `K + 1` runs of the session. Run it once
with `node_mask` all ones, then once per channel `k` with column `k` of `node_mask` set to
zero; the deviation for channel `k` is the unmasked `graph_output` minus the masked one.
This matches the PyTorch result to within 1e-4.

## Verifying a consumer

`fixtures.json` lets an independent implementation check itself without Python:

- `inputs` and `expected`: feed `inputs` (each with `data`, `dims` and `type`) to the
  session and compare the outputs with `expected`. Differences of up to about 1e-5 are
  normal; float arithmetic differs slightly between runtimes.
- `featurization`: for each example molecule, its SMILES and the `node_attributes`,
  `edge_indices` and `edge_attributes` Python computed. A featurizer has to reproduce these
  exactly.

The JavaScript tests in `js/` check `megan.js` — the featurizer, `toFeeds` and
`MeganModel` — and the model against any export folder:

```bash
cd js && npm install
MEGAN_FIXTURES=/path/to/onnx_export npm test
```

If it reports skipped tests, the folder has no `fixtures.json`.

## Limitations

- **Molecules without bonds** (such as methane) cannot be processed. This is a limitation
  of MEGAN itself, not of the export; the exporter and `megan.js` refuse them with a
  message saying so.
- **`MeganEnsemble`** cannot be exported. Export each member model and combine their
  outputs yourself.
- **WebGPU** has not been tested; only the WASM backend has.
- **The CUDA provider** of ONNX Runtime uses TF32 matmuls by default, which puts its output
  about 2e-3 away from the CPU's. In the measurements recorded in ADR-002 the highlighted
  atoms did not change; pass the provider option `use_tf32="0"` when CPU and GPU results
  have to agree tightly.
- **Featurization fidelity.** `megan.js` reproduces the Python featurization exactly on
  the 1,200-molecule validation sample (`scripts/validate_js_featurizer.py` with
  `js/validate_broad.mjs`) and on the hand-picked edge cases in the test fixtures. Models
  trained on a precomputed visual graph dataset see the features stored when that dataset
  was built, which can differ from a fresh featurization: on AqSolDB, for 6 of 9,268
  molecules, all exotic metal oxides.

## Troubleshooting

| Error | Cause |
|---|---|
| `The model takes 38 node and 11 edge features, but the processing has 45 node and 11 edge features.` | The processing is not the one the model was trained with — typically a portable model exported with the full `process.py`. The message names the portable processing when the sizes match it. |
| `Exporting to ONNX needs the "export" extra, which provides onnxscript` | Install the extra: `pip install "graph_attention_student[export]"`. |
| `Example graph 0 has no edges` | An example molecule has no bonds. Pick others. |
| `featurization.json has specVersion …, but this megan.js reads version …` or `featurization.json asks for the atom property "…", which this megan.js does not implement` | The `megan.js` comes from another package version than the export. Use the copy in the export folder, or the CDN URL with the version the message names. |
| `This export has no featurization contract` | The model was trained on features a browser cannot compute; see [Models trained on the full feature set](#manually-from-a-checkpoint). |
| `"onnx_export/" is not an absolute URL` | `MeganModel.load` outside a browser. Pass an `http://` URL, or build the model with the constructor, as in [In Node](#in-node). |
| `Failed to load module script: … the server responded with a MIME type of "text/plain"` (in Chrome) | `megan.js` was linked from raw.githubusercontent.com. Use the jsDelivr URL or the copy in the export folder. |
| `toFeeds needs the model's number of explanation channels` | Pass `manifest.architecture.config.num_channels` as the third argument. |
| `Module.MountedFiles is not available` (in the browser) | The model uses an external-data sidecar, which ONNX Runtime Web cannot load. The exporter never writes one, so the file was re-saved elsewhere; export it again. |
