# ADR-001: Random Access Data Stores for Efficient Data Loading

**Date:** 2026-01-22
**Status:** Accepted

## Context

The existing `SmilesDataset` class uses a streaming/iterable approach with Polars lazy evaluation. While memory-efficient for very large datasets, this approach has limitations:

1. **No random access** - Cannot efficiently access individual samples by index
2. **Shuffling limitations** - Relies on reservoir sampling rather than true shuffling
3. **Validation/test sets** - Streaming is suboptimal when you need to iterate multiple times over the same data
4. **PyTorch DataLoader compatibility** - `IterableDataset` has different multi-worker semantics than map-style datasets

We needed a data loading architecture that:
- Supports efficient random access to individual samples
- Works correctly with PyTorch's multi-worker DataLoader
- Separates concerns between raw data storage, graph processing, and batching
- Handles both CSV-based SMILES data and Visual Graph Dataset (VGD) formats

## Decision

We implemented a **three-layer architecture** for data loading:

```
┌─────────────────────────────────────────────────────────────┐
│                   GraphDataLoader (PyG)                     │
│              Takes any Sequence[GraphDict]                  │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
         ┌────────────────────┴────────────────────┐
         │                                         │
┌────────┴────────┐                    ┌───────────┴───────────┐
│ SmilesGraphStore│                    │VisualGraphDatasetStore│
│Sequence[GraphDict]                   │  Sequence[GraphDict]  │
└────────┬────────┘                    └───────────────────────┘
         │
┌────────┴────────┐
│   SmilesStore   │
│ Sequence[dict]  │
│ (SQLite backend)│
└─────────────────┘
```

### Layer 1: Raw Data Stores

**SmilesStore** (`Sequence[dict]`)
- SQLite-backed storage for CSV data
- `from_csv()` classmethod converts CSV to SQLite (overwrites if exists)
- Returns raw row data as dictionaries with all columns
- Uses `threading.local()` for process-safe database connections

**Why SQLite?**
- Provides O(1) random access via primary key lookup
- No need to load entire dataset into memory
- Battle-tested concurrency support for read operations
- Single file, no external database server required

### Layer 2: Graph Stores

Both implement `Sequence[GraphDict]` - a common interface for the DataLoader.

**SmilesGraphStore**
- Wraps `SmilesStore` + `Processing` instance
- Converts SMILES → GraphDict on-the-fly (no caching)
- User specifies which columns are targets via `target_columns` parameter

**VisualGraphDatasetStore**
- Direct access to VGD directory structure
- Auto-discovers indices from `*.json` files on initialization
- Direct index mapping: `store[5]` reads `5.json`
- Converts JSON lists back to numpy arrays

### Layer 3: DataLoader

**GraphDataLoader** (subclass of `torch_geometric.loader.DataLoader`)
- Accepts any `Sequence[GraphDict]`
- Internally wraps with `_GraphSequenceDataset` for PyG compatibility
- Converts GraphDict → PyG Data via existing `data_from_graph()` function
- Standard PyG batching via `Batch.from_data_list()`

### Multi-Worker Support

SQLite connections cannot be shared across processes. We solve this with **lazy, thread-local connections**:

```python
self._local = threading.local()

@property
def _connection(self):
    if not hasattr(self._local, 'conn') or self._local.conn is None:
        # Each worker creates its own read-only connection
        uri = f'file:{self.sqlite_path}?mode=ro'
        self._local.conn = sqlite3.connect(uri, uri=True)
    return self._local.conn
```

Each DataLoader worker gets its own isolated connection on first access.

### Error Handling Philosophy

Invalid data (e.g., unparseable SMILES, missing VGD indices) raises errors rather than being silently filtered. The user is responsible for cleaning data before training. This makes debugging easier and prevents hidden data loss.

## Consequences

### Positive

1. **Random access** - Can efficiently access any sample by index
2. **True shuffling** - PyTorch DataLoader handles shuffling properly with map-style datasets
3. **Separation of concerns** - Clear layers for storage, processing, and batching
4. **Reusability** - Same `GraphDataLoader` works with both SMILES and VGD data
5. **Multi-worker safe** - Proper handling of SQLite connections across processes
6. **No caching overhead** - Processing happens on-the-fly, memory usage stays constant

### Negative

1. **Initial conversion cost** - `SmilesStore.from_csv()` must process entire CSV once to create SQLite
2. **Disk space** - SQLite file duplicates data from CSV (though typically similar size)
3. **Processing overhead** - SMILES → GraphDict conversion happens every epoch (no caching)

### Neutral

1. **Coexists with SmilesDataset** - Streaming approach still available for very large datasets where random access isn't needed

## Usage Example

```python
from graph_attention_student.torch.data import (
    SmilesStore, SmilesGraphStore, VisualGraphDatasetStore, GraphDataLoader
)
from visual_graph_datasets.processing.molecules import MoleculeProcessing

# From CSV
store = SmilesStore.from_csv('molecules.csv', 'molecules.sqlite')
graph_store = SmilesGraphStore(
    smiles_store=store,
    processing=MoleculeProcessing(),
    target_columns=['solubility'],
    smiles_column='smiles'
)
train_loader = GraphDataLoader(graph_store, batch_size=32, shuffle=True, num_workers=4)

# From VGD directory
vgd_store = VisualGraphDatasetStore('/path/to/dataset')
test_loader = GraphDataLoader(vgd_store, batch_size=32, shuffle=False)

# Training loop
for batch in train_loader:
    predictions = model(batch)
    loss = criterion(predictions, batch.y)
    ...
```

## Alternatives Considered

1. **In-memory caching of processed graphs** - Rejected due to memory constraints for large datasets
2. **HDF5 instead of SQLite** - SQLite is simpler and sufficient for row-based access patterns
3. **Pre-process all SMILES to GraphDict and store** - Would require re-processing if `Processing` parameters change
4. **Lazy filtering of invalid entries** - Rejected in favor of explicit errors to avoid hidden bugs
