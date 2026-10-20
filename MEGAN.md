# MEGAN - Multi-Explanation Graph Attention Network

Reference guide for the `graph_attention_student` Python package. This document covers model instantiation, data preparation, training, inference, checkpoint management, explanation extraction, and advanced features.

## Package Overview

MEGAN is a self-explaining graph neural network for graph-level property prediction (regression or classification). It produces multiple explanation channels that highlight which nodes/edges of a graph are responsible for the prediction. The primary domain is molecular property prediction using SMILES strings, but the architecture is domain-agnostic.

**Key dependencies** (from pyproject.toml):
- `torch>=2.1.2,<2.5.1`
- `torch_geometric>=2.4.0`
- `lightning>=2.1.3` (PyTorch Lightning)
- `visual_graph_datasets>=0.17.0`

**Key imports:**

```python
# Top-level exports (graph_attention_student.__init__)
from graph_attention_student import Megan, MeganEnsemble
from graph_attention_student import data_from_graph, data_list_from_graphs, SmilesDataset

# These are NOT top-level exports — must use full submodule path
from graph_attention_student.torch.data import (
    SmilesStore, SmilesGraphStore, GraphDataLoader, VisualGraphDatasetStore
)
from graph_attention_student.torch.advanced import (
    megan_prediction_report, explain_value, explain_graph
)
```

---

## 1. Graph Representation

MEGAN uses two graph representations interchangeably:

### GraphDict (dict)

The native format from the `visual_graph_datasets` library. A plain Python dict:

```python
graph = {
    'node_attributes': np.ndarray,   # (N, F_node) - node feature matrix
    'edge_attributes': np.ndarray,   # (E, F_edge) - edge feature matrix
    'edge_indices':    np.ndarray,   # (E, 2)      - edge list as (source, target) pairs
    'graph_labels':    list[float],  # optional - target values
    'node_importances': np.ndarray,  # optional (N, K) - ground truth node explanations
    'edge_importances': np.ndarray,  # optional (E, K) - ground truth edge explanations
    'node_coordinates': np.ndarray,  # optional (N, 2|3) - spatial positions
    'graph_weight':     float,       # optional - sample weight for training
}
```

Note: `edge_indices` shape is **(E, 2)** in GraphDict format — each row is a `(source, target)` pair. PyG expects **(2, E)** (first row = all sources, second row = all targets). The `data_from_graph` function handles this transpose automatically; do not transpose manually.

### Constructing a Custom GraphDict (Non-SMILES)

For non-molecular or custom graph data, construct a GraphDict manually:

```python
import numpy as np

graph = {
    'node_attributes': np.random.randn(5, 3).astype(np.float32),  # 5 nodes, 3 features each
    'edge_attributes': np.random.randn(8, 2).astype(np.float32),  # 8 edges, 2 features each
    'edge_indices': np.array([                                      # (E, 2) edge list
        [0, 1], [1, 0],  # undirected edges should include both directions
        [1, 2], [2, 1],
        [2, 3], [3, 2],
        [3, 4], [4, 3],
    ]),
    'graph_labels': [1.5],  # target value(s)
}
```

All feature values must be float32. Graphs should have at least 1 node and 1 edge. For undirected graphs, include both edge directions `(i, j)` and `(j, i)`.

### PyG Data (torch_geometric.data.Data)

```python
Data(
    x=Tensor,           # (N, F_node)  - node features
    edge_attr=Tensor,   # (E, F_edge)  - edge features
    edge_index=Tensor,  # (2, E)       - COO format (source row, target row)
    y=Tensor,           # target values
    batch=Tensor,       # (N,) - graph assignment index per node (for batching)
)
```

### Converting Between Formats

```python
from graph_attention_student import data_from_graph, data_list_from_graphs

# Single graph
data: Data = data_from_graph(graph_dict)

# List of graphs
data_list: list[Data] = data_list_from_graphs(list_of_graph_dicts)
```

`data_from_graph` transposes `edge_indices` from (E, 2) to (2, E), converts arrays to tensors (float32), and attaches optional fields (`node_importances`, `edge_importances`, `pos`, `train_weight`) when present.

---

## 2. Creating Graphs from SMILES

For molecular graphs, use `MoleculeProcessing` from `visual_graph_datasets`:

```python
from visual_graph_datasets.processing.molecules import MoleculeProcessing

processing = MoleculeProcessing()

# Convert a SMILES string to a GraphDict
graph: dict = processing.process('CCO')

# The processing instance knows the feature dimensions
node_dim: int = processing.get_num_node_attributes()
edge_dim: int = processing.get_num_edge_attributes()
```

The `processing` object is essential - it defines the featurization scheme and must be the same instance (or equivalent) used for both training and inference.

---

## 3. Model Instantiation

```python
from graph_attention_student import Megan

model = Megan(
    # --- Encoder ---
    node_dim=9,                   # number of input node features
    edge_dim=4,                   # number of input edge features
    units=[64, 64, 64],           # hidden dims per graph encoder layer (len = num layers)
    hidden_units=128,             # hidden dim inside each attention layer's MLPs
    layer_version='v2',           # 'v1', 'v2', or 'v3' attention layer variant
    encoder_dropout_rate=0.0,     # dropout on node embeddings between encoder layers

    # --- Explanations ---
    num_channels=2,               # number of explanation channels
                                  #   regression: must be 2 (negative + positive)
                                  #   classification: must equal number of classes
    importance_mode='regression', # None, 'regression', or 'classification'
                                  #   None disables explanation co-training
    importance_factor=1.0,        # weight of explanation co-training loss (0 = disabled)
    importance_offset=0.1,        # controls explanation coverage (lower = more focused, default 0.1)
    sparsity_factor=0.5,          # weight of Hoyer sparsity regularization on explanations
    attention_aggregation='max',  # how to aggregate attention across layers: 'sum'|'mean'|'max'|'min'
    importance_target='node',     # 'node' or 'edge' - level of explanation approximation
    projection_units=[64],        # per-channel projection MLP after graph pooling

    # --- Prediction ---
    final_units=[64, 32, 1],      # final MLP dims; LAST value = output dimension
                                  #   regression: typically 1
                                  #   classification: number of classes
    prediction_mode='regression', # 'regression', 'classification', or 'bce'
    prediction_factor=1.0,        # weight of prediction loss

    # --- Training ---
    learning_rate=1e-4,

    # --- Optional ---
    normalize_embedding=False,    # L2-normalize graph embeddings (for contrastive learning)
    contrastive_factor=0.0,       # weight of contrastive representation loss (0 = disabled)
    fidelity_factor=0.0,          # weight of fidelity loss (0 = disabled)
    final_dropout_rate=0.0,       # dropout before final prediction layer
    class_weights=None,           # list of per-class weights for classification
    output_norm=None,             # if set, L2-normalizes output logits to this radius
    label_smoothing=0.0,          # label smoothing for classification targets
)
```

### Key Constraints

- **Regression with explanations**: `num_channels` must be exactly 2. Channel 0 = "negative" contribution, channel 1 = "positive" contribution.
- **Classification with explanations**: `num_channels` must equal the number of classes (= `final_units[-1]`).
- **`importance_factor=0`**: Disables explanation co-training entirely (no explanation loss).
- The model outputs **logits** for classification (not probabilities). Apply `softmax` manually if needed.

### Prediction Modes

- **`'regression'`**: Continuous value prediction. Loss: MSE. Output: raw values in the same scale as training targets. Uses 2 explanation channels (negative/positive).
- **`'classification'`**: Multi-class classification. Loss: EvidentialLoss (Dirichlet-based). Output: class logits (apply `torch.softmax(logits, dim=-1)` for probabilities). `num_channels` must equal number of classes.
- **`'bce'`**: Binary cross-entropy mode for multi-label or binary classification. Loss: `BCEWithLogitsLoss`. Output: per-class logits (apply `torch.sigmoid(logits)` for probabilities). Useful when multiple classes can be active simultaneously, unlike `'classification'` which assumes mutually exclusive classes.

### Regression Reference (Auto-Centering)

During regression training, the model maintains a **running mean** of target values across all batches. This running mean is stored as `regression_reference` and added to the model's raw output as a bias term, effectively centering predictions around the training distribution. This is a non-trainable parameter (`requires_grad=False`) that is saved with the checkpoint and restored on load.

---

## 4. Forward Pass

### Low-Level: PyG Data (returns torch Tensors)

```python
# Single Data object (or a batched Data from DataLoader)
info: dict = model(data)   # calls model.forward(data), returns TORCH TENSORS
```

**Returned dict keys:**

| Key | Shape | Description |
|-----|-------|-------------|
| `graph_output` | (B, O) | Final predictions (logits for classification) |
| `graph_embedding` | (B, D, K) | Per-channel graph embeddings |
| `node_embedding` | (B*V, D, K) | Per-channel node embeddings from last encoder layer |
| `node_importance` | (B*V, K) | Raw node importance values |
| `node_importance_norm` | (B*V, K) | Normalized node importance (max=1 per graph, thresholded) |
| `edge_importance` | (B*E, K) | Raw edge importance values |
| `edge_importance_norm` | (B*E, K) | Normalized edge importance |
| `graph_variance` | (B, 1) | Predicted variance (only if MVE is active) |

Where B = batch size, V = total nodes across batch, E = total edges across batch, D = embedding dim, K = num_channels, O = output dim.

### High-Level: GraphDict Convenience Methods (returns numpy arrays)

These methods handle conversion from GraphDicts to PyG Data internally. Unlike `forward()`, they return **numpy arrays** with batch dimensions removed (each result dict corresponds to one graph):

```python
# Single graph -> full info dict (numpy arrays, unbatched)
info: dict = model.forward_graph(graph_dict)
# info['graph_output'] -> np.ndarray of shape (O,)
# info['node_importance'] -> np.ndarray of shape (N, K)

# Multiple graphs -> list of info dicts (one per graph)
results: list[dict] = model.forward_graphs(graphs, batch_size=1000)

# Just the prediction values (no explanations)
predictions: np.ndarray = model.predict_graphs(graphs, batch_size=1000)
# shape: (B, O)

# Single prediction
prediction: np.ndarray = model.predict_graph(graph_dict)
# shape: (O,)
```

### Eval Mode and Device

- **Eval mode**: Always call `model.eval()` before inference. This is critical because the model uses BatchNorm layers which behave differently during training vs evaluation. Note: `Megan.load()` automatically calls `model.eval()`, so you only need to call it explicitly if you trained the model in the same session.
- **Device management**: MEGAN inherits from `pl.LightningModule`, so use standard PyTorch device management:

```python
model = model.to('cuda')    # move to GPU
model = model.to('cpu')     # move to CPU
# During training, pl.Trainer handles device placement automatically via accelerator='auto'
```

---

## 5. Data Loading

### Which Data Loader to Use?

| Option | Best for | Shuffling | Random access | Memory |
|--------|----------|-----------|---------------|--------|
| **SmilesDataset** | Training (large CSVs) | Reservoir sampling | No (streaming) | Low |
| **SmilesStore + SmilesGraphStore** | Validation/test, or when you need indexed access | Via DataLoader `shuffle=True` | Yes | Low (SQLite-backed) |
| **VisualGraphDatasetStore** | Pre-processed VGD format datasets | Via DataLoader `shuffle=True` | Yes | Low (JSON files) |

### Option A: SmilesDataset (Streaming / IterableDataset)

For training from CSV files. Streams SMILES, converts to graphs on-the-fly. Supports reservoir sampling for shuffling.

**Warning**: Invalid SMILES strings are **silently skipped** during iteration (no error, no log message). Your effective dataset size may be smaller than the CSV row count.

```python
from graph_attention_student import SmilesDataset
from torch_geometric.loader import DataLoader

dataset = SmilesDataset(
    dataset='molecules.csv',          # CSV path, DataFrame, or list of dicts
    smiles_column='smiles',           # column with SMILES strings
    target_columns=['solubility'],    # list of target column names
    processing=MoleculeProcessing(),  # featurization
    reservoir_sampling=True,          # enables shuffle via reservoir
    reservoir_size=1000,              # reservoir buffer size
)

loader = DataLoader(
    dataset,
    batch_size=64,
    drop_last=True,
    num_workers=4,
    prefetch_factor=2,
)
```

### Option B: SmilesStore + SmilesGraphStore (Random Access / SQLite-backed)

For random-access loading (required for validation/test sets or when you need indexed access):

**Warning**: `SmilesGraphStore` returns `None` for invalid SMILES (with a warning log). The `GraphDataLoader` will raise an error if it encounters `None` — filter invalid entries beforehand.

```python
from graph_attention_student.torch.data import SmilesStore, SmilesGraphStore, GraphDataLoader

# Step 1: Create SQLite store from CSV (one-time, overwrites if exists)
store = SmilesStore.from_csv('molecules.csv', 'molecules.sqlite')

# Step 2: Wrap with graph conversion
graph_store = SmilesGraphStore(
    smiles_store=store,
    processing=MoleculeProcessing(),
    target_columns=['solubility'],
    smiles_column='smiles',
)

# Step 3: Use with DataLoader
loader = GraphDataLoader(graph_store, batch_size=32, shuffle=True, num_workers=4)
```

`SmilesStore` uses thread-local SQLite connections, making it safe for multi-worker DataLoaders.

### Option C: VisualGraphDatasetStore (VGD format)

For datasets stored in Visual Graph Dataset format (directory of JSON files):

```python
from graph_attention_student.torch.data import VisualGraphDatasetStore, GraphDataLoader

vgd_store = VisualGraphDatasetStore('/path/to/vgd_dataset')
loader = GraphDataLoader(vgd_store, batch_size=32, shuffle=True)
```

---

## 6. Training

MEGAN is a PyTorch Lightning module. Training uses `pl.Trainer`:

```python
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from visual_graph_datasets.processing.molecules import MoleculeProcessing

processing = MoleculeProcessing()

# Create model
model = Megan(
    node_dim=processing.get_num_node_attributes(),
    edge_dim=processing.get_num_edge_attributes(),
    units=[64, 64, 64],
    final_units=[64, 32, 1],
    prediction_mode='regression',
    importance_mode='regression',
    importance_factor=1.0,
    sparsity_factor=0.5,
    learning_rate=1e-4,
)

# Create data loader (see Section 5)
loader_train = DataLoader(dataset, batch_size=64, drop_last=True)

# Train
trainer = pl.Trainer(
    max_epochs=200,
    accelerator='auto',
    devices='auto',
    enable_progress_bar=True,
    logger=False,
)
trainer.fit(model, train_dataloaders=loader_train)
```

### Training with Validation

```python
# Create a separate validation loader (use SmilesStore for random access)
loader_val = GraphDataLoader(val_graph_store, batch_size=64, shuffle=False)

trainer.fit(model, train_dataloaders=loader_train, val_dataloaders=loader_val)
```

Note: The `Megan` class does not implement `validation_step` by default. To use validation, subclass `Megan` and implement it, or rely on training loss monitoring.

### Training Loss Components

The total loss is a weighted sum controlled by the corresponding `*_factor` parameters:

```
loss = prediction_factor * loss_pred
     + importance_factor * loss_expl
     + sparsity_factor   * loss_spar
     + contrastive_factor * loss_cont
     + fidelity_factor   * loss_fid
```

| Loss | Factor param | Description |
|------|-------------|-------------|
| `loss_pred` | `prediction_factor` | MSE (regression), EvidentialLoss (classification), or BCEWithLogitsLoss (bce) |
| `loss_expl` | `importance_factor` | Explanation co-training (see below) |
| `loss_spar` | `sparsity_factor` | Hoyer square regularization: promotes sparse explanations |
| `loss_cont` | `contrastive_factor` | SimCLR-style contrastive loss on per-channel graph embeddings |
| `loss_fid` | `fidelity_factor` | Promotes positive fidelity (see below) |

Set any factor to `0.0` to disable that loss component. A minimal configuration would set only `prediction_factor > 0` and `importance_factor > 0`.

**Explanation co-training loss (`loss_expl`)**: This does NOT use MSE on the target. Instead, it converts the prediction task into a binary classification problem on the explanation masks. For regression, target values are split at the batch median into "low" (channel 0) and "high" (channel 1) groups. The pooled importance values for each channel are then trained with BCE loss to predict this binary assignment. This promotes channel 0 to activate on graphs with low target values and channel 1 on graphs with high target values. For classification, each channel's pooled importance is trained to predict its corresponding class via BCE.

**Fidelity loss (`loss_fid`)**: Performs a leave-one-out forward pass for each channel (masking that channel's importance to zero) and computes the prediction deviation. The loss has two terms: `mean(ReLU(-fidelity)) + 0.1 * mean(fidelity^2)` — it penalizes negative fidelity (channel not contributing in expected direction) while also mildly regularizing fidelity magnitude.

### Optimizer

Uses Adam by default. Optionally supports cyclic learning rate scheduling via `lr_scheduler='cyclic'`.

---

## 7. Saving and Loading Checkpoints

### Save

```python
model.save('model.ckpt')
```

Saves a torch checkpoint containing: `state_dict`, `hyper_parameters` (constructor args for reconstruction), `pytorch-lightning_version`, and package `version`.

### Load

```python
model = Megan.load('model.ckpt')
# model is already in eval mode after loading
```

The `load` classmethod reconstructs the model from stored hyperparameters. If loading fails due to a version mismatch, the error message will include the model's version and suggest the correct package version to install.

### Saving/Loading the Processing Module

The processing instance must be saved alongside the model for reproducible inference:

```python
from visual_graph_datasets.processing.base import create_processing_module
from visual_graph_datasets.util import dynamic_import

# Save
processing_content = create_processing_module(processing)
with open('process.py', 'w') as f:
    f.write(processing_content)

# Load
module = dynamic_import('process.py')
processing = module.processing
```

---

## 8. Explanation System

### Channel Semantics

- **Regression** (2 channels):
  - Channel 0: "Negative" - highlights substructures that decrease the predicted value
  - Channel 1: "Positive" - highlights substructures that increase the predicted value

- **Classification** (K channels, K = num classes):
  - Channel k highlights substructures that support class k

### Extracting Explanations

```python
model.eval()
info = model.forward_graph(graph)

node_importance = info['node_importance']       # (N, K) raw
node_importance_norm = info['node_importance_norm']  # (N, K) normalized [0, 1]
edge_importance = info['edge_importance']       # (E, K) raw
edge_importance_norm = info['edge_importance_norm']  # (E, K) normalized [0, 1]
```

Normalized importances are per-graph max-normalized (max = 1.0) with values below 0.1 scaled down by 0.01 (soft thresholding).

### How Attention Becomes Explanation

1. Each graph encoder layer produces edge attention logits `(B*E, K)` — one value per edge per channel.
2. After all layers, logits are stacked `(B*E, K, L)` where L = number of layers. **Sigmoid is applied first** to each layer's logits (converting to [0,1] range), then the results are reduced across layers via `attention_aggregation` ('sum', 'mean', 'max', 'min').
3. This yields edge importance `(B*E, K)` — attention values in [0, 1] per edge per channel.
4. Node importance `(B*V, K)` is derived from edge importance by averaging the max-pool from source endpoints and destination endpoints: `0.5 * (max_pool_by_source + max_pool_by_target)`.
5. Both node and edge importance are **normalized per graph**: divided by the max value across all channels within each graph. Values below 0.1 are soft-thresholded (multiplied by 0.01) to suppress noise. A safeguard prevents division by near-zero max values.
6. The normalized node importance acts as weights in the global sum pooling step: `graph_embedding_k = sum(node_embedding_k * node_importance_norm_k)`. Each channel's graph embedding therefore only contains information from its highlighted subgraph.

### Fidelity / Leave-One-Out Analysis

Measures how much each channel actually affects the prediction:

```python
# deviations: np.ndarray of shape (B, O, K)
# deviations[i, o, k] = prediction[i,o] - prediction_without_channel_k[i,o]
deviations = model.leave_one_out_deviations(graphs, batch_size=1000)
```

---

## 9. Visualization and Reports

### Prediction Report (PDF)

Generates a comprehensive PDF with prediction, explanation heatmaps, fidelity analysis, and model stats:

```python
from graph_attention_student.torch.advanced import megan_prediction_report

html = megan_prediction_report(
    value='CCO',                 # SMILES string
    model=model,                 # trained Megan instance
    processing=processing,       # MoleculeProcessing instance
    output_path='report.pdf',    # output PDF path
    vis_width=1000,              # visualization image width in pixels
    vis_height=1000,             # visualization image height in pixels
    unit='mg/L',                 # optional unit string shown with prediction
)
```

### Explanation Visualization (matplotlib)

```python
from graph_attention_student.torch.advanced import explain_value, explain_graph

# From SMILES string
fig, axs, info, graph = explain_value(
    value='CCO',
    model=model,
    processing=processing,
    strategy='joint',       # 'joint' (all channels one plot) or 'split' (separate subplots)
    fig_size=10,
    radius=50,              # node circle radius
    thickness=20,           # edge line thickness
)

# From existing GraphDict + image
fig, axs, info = explain_graph(
    graph=graph_dict,
    image_path='/path/to/graph_image.png',
    model=model,
    strategy='split',
)
```

---

## 10. MeganEnsemble

Combines multiple trained Megan models for improved predictions and uncertainty estimation:

```python
from graph_attention_student import Megan, MeganEnsemble

models = [Megan.load(f'model_{i}.ckpt') for i in range(5)]
ensemble = MeganEnsemble(
    models=models,
    combine_importances_strategy='mean',  # 'mean', 'discount', 'min', 'max'
)
```

### Ensemble Interface

The ensemble has the same interface as a single Megan model (`forward_graph`, `forward_graphs`, `predict_graphs`), plus:

```python
# Forward pass returns additional keys:
info = ensemble.forward_graph(graph)
info['graph_output']          # mean prediction across models
info['graph_uncertainty']     # std dev across model predictions (uncertainty estimate)
info['graph_output_stack']    # individual model predictions
info['node_importance']       # combined node explanations
info['edge_importance']       # combined edge explanations

# Uncertainty estimation
uncertainties: np.ndarray = ensemble.estimate_uncertainty_graphs(graphs)
```

### Ensemble Save/Load

**Important**: Unlike `Megan.save()` which creates a single `.ckpt` file, `MeganEnsemble.save()` **creates a directory** containing an `ensemble.ckpt` metadata file and individual `model_0.ckpt`, `model_1.ckpt`, etc. files. The directory must not already exist.

```python
# Save (creates a NEW directory — will error if it already exists)
ensemble.save('ensemble_dir')

# Load (pass the directory path)
ensemble = MeganEnsemble.load('ensemble_dir')
```

### Uncertainty Calibration

```python
# Calibrate uncertainty estimates using isotonic regression
ensemble.calibrate_uncertainty(graphs_val, errors_val)

# Stack predictions using a meta-learner
ensemble.stack_predictions(graphs_val, values_val)
```

---

## 11. Mean Variance Estimation (MVE)

MEGAN supports predicting aleatoric uncertainty via a variance head:

```python
from graph_attention_student.torch.megan import MveCallback

# The MVE head is always constructed but inactive by default.
# Use a callback to activate it after warmup:
mve_callback = MveCallback(warmup_epochs=100)

trainer = pl.Trainer(
    max_epochs=300,
    callbacks=[mve_callback],
)
trainer.fit(model, train_dataloaders=loader)

# After training, forward pass includes variance:
info = model.forward_graph(graph)
variance = info['graph_variance']  # (1,) predicted variance
```

The MVE head trains with a Negative Log-Likelihood loss that replaces the standard prediction loss after `warmup_epochs`.

---

## 12. CLI

The package provides a `megan` CLI (installed via the package entry point):

```bash
# Train a model from CSV
megan train data.csv \
    --smiles-column smiles \
    --target-columns value \
    --prediction-mode regression \
    --units 64,64,64 \
    --final-units 64,32,1 \
    --max-epochs 200 \
    --batch-size 64 \
    --importance-factor 1.0 \
    --sparsity-factor 0.5 \
    --output-path model.ckpt

# Generate prediction report
megan predict "CCO" \
    --model-path model.ckpt \
    --processing-path process.py \
    --output-path report.pdf
```

The `train` command saves both `model.ckpt` and `process.py` (the processing module).

---

## 13. Experiment Framework

The repository includes a set of experiment scripts built on PyComex (a microframework for reproducible ML experiments). These are located in `graph_attention_student/experiments/`:

- `vgd_torch__megan.py` - Base MEGAN training experiment
- `vgd_torch__megan__aqsoldb.py` - AqSolDB solubility dataset
- `vgd_torch__megan__mutagenicity.py` - Mutagenicity classification

Experiments inherit from the base and override parameters at module level. See separate PyComex documentation for details on running and extending experiments.

---

## 14. Architecture Internals

### Layer Structure

```
Input: node_attributes (N, F_node), edge_attributes (E, F_edge), edge_index (2, E)
  |
  v
[Linear Embedding] node_dim -> units[0]
  |
  v  (replicated K times for K channels)
[ParallelHeadAttention layers] x len(units)
  Each contains K independent GraphAttentionLayerV2 instances
  Each layer: node_emb (B*V, units[i], K), edge_attention_logits (B*E, K)
  Dropout applied to node embeddings after each layer (encoder_dropout_rate)
  |
  v
[Attention Aggregation] per-layer sigmoid -> stack -> aggregate across layers
  -> edge_importance (B*E, K)
  -> node_importance (B*V, K) via max-pool from edges
  -> normalize per graph, soft-threshold at 0.1
  |
  v
[Importance-Weighted Sum Pooling] per channel
  node_emb_k * node_importance_norm_k -> sum -> graph_emb_k (B, D)
  |
  v
[Per-Channel Projection MLP] K independent MLPs
  Each: Linear -> SiLU -> LayerNorm (per projection_units entry)
  graph_emb_k -> projected_emb_k (B, proj_dim)
  Optional L2 normalization (if normalize_embedding=True)
  |
  v
[Concatenation] all channels -> (B, proj_dim * K)
  |
  v
[Final Prediction MLP] defined by final_units
  Intermediate: Linear -> SiLU -> LayerNorm + dropout (final_dropout_rate)
  Last layer: Linear (no activation)
  + regression_reference bias added (regression only)
  -> (B, output_dim)
```

### GraphAttentionLayerV2

The default attention layer (`layer_version='v2'`). Each layer's message passing:

1. Computes neighbor sum: `x_neighbor = SumNeighbors(x, edge_index)`
2. For each edge (i, j), constructs message input: `[x_i, x_j, x_neighbor_j, edge_attr]`
3. Attention: `logit = MLP_attention(message_input)` (3-layer MLP with BatchNorm + LeakyReLU), `alpha = sigmoid(logit)`
4. Message: `msg = alpha * MLP_message(message_input)` (2-layer MLP with BatchNorm + LeakyReLU)
5. Aggregation: `node_emb = sum(messages_to_node)`
6. Transform + Residual: `output = MLP_transform([node_emb, x]) + x` (2-layer MLP with BatchNorm + LeakyReLU)

Returns `(node_embedding, attention_logits)`.

**Other layer versions**: `'v1'` uses simpler single-linear attention and message networks. `'v3'` uses multiple message heads (default 3) averaged together, with tanh on the output.

---

## 15. Common Patterns

### Full Training + Inference Pipeline

```python
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.processing.base import create_processing_module
from graph_attention_student import Megan, SmilesDataset
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl

# 1. Setup processing
processing = MoleculeProcessing()

# 2. Create dataset and loader
dataset = SmilesDataset(
    dataset='train.csv',
    smiles_column='smiles',
    target_columns=['value'],
    processing=processing,
)
loader = DataLoader(dataset, batch_size=64, drop_last=True, num_workers=4)

# 3. Create model
model = Megan(
    node_dim=processing.get_num_node_attributes(),
    edge_dim=processing.get_num_edge_attributes(),
    units=[64, 64, 64],
    final_units=[64, 32, 1],
    prediction_mode='regression',
    importance_mode='regression',
    importance_factor=1.0,
    sparsity_factor=0.5,
    learning_rate=1e-4,
)

# 4. Train
trainer = pl.Trainer(max_epochs=200, accelerator='auto')
trainer.fit(model, train_dataloaders=loader)

# 5. Save
model.eval()
model.save('model.ckpt')
with open('process.py', 'w') as f:
    f.write(create_processing_module(processing))

# 6. Load and predict
from visual_graph_datasets.util import dynamic_import

model = Megan.load('model.ckpt')
processing = dynamic_import('process.py').processing

graph = processing.process('c1ccccc1')
info = model.forward_graph(graph)
prediction = info['graph_output']           # shape (O,)
node_imp = info['node_importance']          # shape (N, K)
edge_imp = info['edge_importance']          # shape (E, K)
```

### Classification Setup

```python
model = Megan(
    node_dim=processing.get_num_node_attributes(),
    edge_dim=processing.get_num_edge_attributes(),
    units=[64, 64, 64],
    num_channels=3,                    # = number of classes
    final_units=[64, 32, 3],           # last value = number of classes
    prediction_mode='classification',
    importance_mode='classification',
    importance_factor=1.0,
)
```

For classification, `forward_graph` returns logits. To get probabilities:

```python
import torch
logits = torch.tensor(info['graph_output'])
probabilities = torch.softmax(logits, dim=-1)
predicted_class = torch.argmax(logits).item()
```

---

## 16. Testing Utilities

The `graph_attention_student.testing` module provides helpers for tests:

```python
from graph_attention_student.testing import get_mock_graphs, model_from_processing

# Generate random test graphs
graphs = get_mock_graphs(
    num_graphs=10,
    num_node_attributes=9,
    num_edge_attributes=4,
)

# Quick model instantiation from a processing instance
model = model_from_processing(processing, prediction_mode='regression')
```

---

## 17. Common Pitfalls

- **Forgetting `model.eval()`**: The model uses BatchNorm layers. Running inference in training mode will produce incorrect results. `Megan.load()` calls `eval()` automatically, but after `trainer.fit()` you must call it manually.
- **SMILES silently skipped**: `SmilesDataset` skips unparseable SMILES without any log output. If your dataset seems smaller than expected, validate SMILES separately.
- **Classification outputs are logits**: The model always outputs raw logits, never probabilities. Apply `softmax` (classification) or `sigmoid` (bce) manually.
- **`forward()` vs `forward_graph()`**: `forward()` returns torch tensors with flattened batch dimensions (B*V, B*E). `forward_graph()` / `forward_graphs()` return numpy arrays split per graph. Use the latter for inference convenience.
- **Ensemble save path**: `MeganEnsemble.save(path)` creates a new directory at `path`. It will fail if the directory already exists. `Megan.save(path)` creates a single file.
- **Training `RuntimeError` silently caught**: If a `RuntimeError` occurs during `training_step` (e.g., shape mismatch from a malformed graph), the error is printed but training continues with a zero loss for that batch. Watch training logs for these messages.
