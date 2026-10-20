# ADR-002: ONNX as the Portability Path for Trained MEGAN Models

**Date:** 2026-08-25
**Status:** Accepted

## Context

A trained MEGAN model could only be run inside this Python stack — PyTorch, PyTorch
Geometric and `visual_graph_datasets`. That made it impossible to ship a model anywhere
else: a static demo site, a JavaScript tool, or any non-Python service. The motivating
case is a static website that predicts a molecular property and renders the explanation
heatmap client-side, with no inference server at all.

Two design documents already targeted this goal by a different route.
`RUST_COMPILE.md` and `WASM_COMPILE.md` (kept on the `compile-design-docs` branch, since
nothing in them is implemented) describe a compiler that extracts the model
architecture, folds BatchNorm into the preceding linear layers, and generates
hand-written Rust which is then built to WebAssembly with `wasm-bindgen`. That approach
is fast — it eliminates Python dispatch, PyG's message-passing plumbing and per-op
tensor allocation, which is where the time actually goes for molecular graphs — but it
is a substantial amount of code to write and to keep correct as the model evolves.

The blocker for any export route is that MEGAN's attention layers subclass
`torch_geometric.nn.MessagePassing`. `propagate()` is generated at runtime by PyG's code
generator, and the layers stash their attention logits on `self` inside `message()` and
read them back afterwards. Neither survives tracing.

## Decision

Add an ONNX export as the **portability** path, alongside — not instead of — the
Rust/WebAssembly **speed** path. ONNX reaches every target the browser case needs at a
fraction of the implementation cost, and ONNX Runtime is a maintained dependency rather
than a code generator we own.

Four decisions inside that:

**1. Explicit gather/scatter stand-ins, not a refactor of the layers.**
`torch/export/layers_explicit.py` reimplements the plumbing of each attention layer as
an explicit gather → elementwise → scatter sequence, wrapping the trained layer and
reusing its submodules by reference. `message()` and `propagate()` are untouched, so the
training path — and the validated 1.6.0 results — cannot be perturbed. The cost is
duplicated arithmetic, which is contained by a parity test that asserts bit-identical
output for every supported layer variant.

**2. Every segment reduction must lower to `ScatterND`.**
ONNX Runtime Web's WebGPU execution provider has **no `ScatterElements` kernel at all**,
and `ScatterElements` is exactly what the dynamo exporter emits for PyG's `scatter` and
for `torch.scatter_reduce_`. `ScatterND` gained `reduction='max'` in opset 18. MEGAN uses
only two reductions — `add` and `max` — because every attention value is a per-edge
sigmoid and there is no segment softmax anywhere in the model. `add` is reachable via
`index_add_`; `max` has no aten spelling that works, so it is a custom operator plus an
onnxscript translation. `export_megan_onnx` fails the export if a `ScatterElements` ever
appears, turning a hard-to-diagnose failure in someone's browser into an export-time
error.

**3. The ONNX boundary sits after featurization.**
The graph takes tensors, not SMILES: RDKit's sanitization and aromaticity perception
have no ONNX equivalent. Instead the exporter derives a machine-readable featurization
contract from the live processing object and ships a reference JavaScript
implementation of it. Because the contract is derived rather than hardcoded, it cannot
drift from the processing the model was trained on — which matters, since the processing
is *not* stored in the checkpoint.

**4. Leave-one-out via an optional `node_mask` input.**
Rather than baking a `num_channels`-long ablation loop into the graph, `node_mask` is an
input and a consumer runs the session K+1 times. This keeps one exported graph valid for
any number of channels.

## Consequences

**Good**

- A trained model runs under ONNX Runtime in Python, Node and the browser on the WASM
  backend, which the tests cover. The graph contains only `ScatterND`, so it should also
  run on the WebGPU backend, but no exported model has been run there yet. Exported
  models are 1–5 MB: 1.1 MB for the `units=[16, 16]` test model, 5.4 MB for the
  `units=[128, 128, 128]` AqSolDB model.
- The export is verified rather than asserted: bit-identical layer parity, PyTorch vs
  onnxruntime numerics across batch sizes, and *task-level* explanation equivalence —
  the same atoms above the threshold and the same importance ranking. The last of these
  matters because the normalization applies a discontinuous
  `where(x < 0.1, x * 0.01, x)`, so a float-level difference can flip a node across the
  threshold and visibly change an explanation that a plain `allclose` would pass.
- `torch/export/architecture.py` is shared with the Rust/WebAssembly path, so the two
  cannot disagree about what a model is.

**Bad / accepted costs**

- The explicit layers duplicate the arithmetic in `message()`. A change to an attention
  layer that is not mirrored there will be caught by the parity test, but it *is* a
  second place to edit.
- `megan.py` gained a small indirection: the six segment reductions in `forward` now go
  through `_segment_max` / `_segment_sum`, and `forward` takes an optional `num_graphs`.
  The default implementations are the previous calls verbatim, and output was verified
  bit-identical across v1/v2/v3 and both the plain and ablation branches.
- `onnx`, `onnxruntime` and `onnxscript` are needed to export. They live in an `export`
  extra rather than the base dependencies.

**Limitations recorded in the manifest**

- Classification models export **raw logits**: `lay_act_final` is an `Identity` and is
  never called in `forward`. A consumer must apply softmax or sigmoid itself.
- Molecules with no bonds (methane) cannot be exported. This is pre-existing —
  `MoleculeProcessing` emits `edge_indices` with shape `(0,)` rather than `(0, 2)` and
  `Megan.forward` raises `IndexError` — but the exporter now rejects them by name
  instead of failing obscurely.
- `MeganEnsemble` is not supported; export the member models separately.
- The model must be exported self-contained. ONNX Runtime Web cannot load an
  external-data sidecar, failing with `Module.MountedFiles is not available`.

## Measured performance

Benchmarked with `scripts/benchmark_onnx.py` on 600 aqsoldb molecules (mean 16.2 atoms),
`units=[64,64,64]`, `num_channels=2`, `final_units=[64,32,1]` — a 3 MB ONNX model.
Forward pass only; featurization and PyG collation are excluded from both sides, and
both runtimes are pinned to the same thread count. Median of 25 runs.

**CPU** (ONNX Runtime 1.29, speedup is PyTorch → ONNX):

| batch | 1 thread | 8 threads |
|---|---|---|
| 1 | 2.24 → 0.42 ms (**5.4x**) | 2.65 → 0.44 ms (**6.0x**) |
| 8 | 6.70 → 3.83 ms (1.8x) | 5.34 → 1.72 ms (3.1x) |
| 32 | 20.4 → 14.4 ms (1.4x) | 10.0 → 5.03 ms (2.0x) |
| 128 | 81.0 → 60.4 ms (1.3x) | 28.8 → 14.7 ms (2.0x) |
| 512 | 399 → 259 ms (1.5x) | 135 → 67.2 ms (2.0x) |

**GPU** (RTX 5070 Ti Laptop; PyTorch timed including the host-to-device copy, since the
ONNX figures are fed from host memory):

| batch | PyTorch | ONNX | speedup |
|---|---|---|---|
| 1 | 3.84 ms | 1.85 ms | 2.1x |
| 32 | 4.42 ms | 1.93 ms | 2.3x |
| 128 | 4.02 ms | 1.49 ms | 2.7x |
| 512 | 10.8 ms | 4.38 ms | 2.5x |

Three things follow from this.

**The single-molecule case is where ONNX wins biggest** — 5–6x, and it is exactly the
case a static site hits. This is consistent with the premise of `WASM_COMPILE.md`: for
graphs this small the cost is Python dispatch, PyG's message-passing plumbing and
per-op allocation, not arithmetic. Removing the framework removes most of the time.

**The GPU is the wrong device for interactive single-molecule inference.** At batch 1
ONNX on the CPU (0.42 ms) beats ONNX on the GPU (1.85 ms) by more than 4x, because both
GPU paths are dominated by kernel-launch overhead — PyTorch sits at ~3.9 ms regardless of
batch size all the way to 128. The GPU only earns its place for screening: at batch 512
it reaches 117k molecules/s against 7.6k on the CPU.

**The Rust/WebAssembly path has a smaller remaining margin than it did before.** ONNX
already captures most of the framework-overhead win. That path is now worth pursuing only
where its specific advantages matter, not for the general speed-up it was motivated by.

One caveat found while measuring: ONNX Runtime's `CUDAExecutionProvider` uses TF32
matmuls by default, putting it ~1.7e-3 from the CPU provider rather than ~2e-6. The
explanation masks were unaffected — the same atoms stayed above the threshold — but
`use_tf32="0"` restores tight agreement for about 20% of the throughput. This is
recorded in the manifest.


## Alternatives considered

**Rust/WebAssembly codegen only.** Faster, but a large amount of owned code for a model
whose forward pass is ~2.2M MACs — fast enough under WASM that the difference is
unlikely to be perceptible for interactive single-molecule use. Kept as a separate path
for cases where throughput genuinely matters.

**Emitting `ScatterElements` and supporting WASM only.** Simpler, but it silently
forecloses WebGPU, and the fix later would mean a second divergent export path.

**Putting SMILES parsing inside the ONNX graph.** Not feasible: aromaticity perception
and sanitization are not expressible as ONNX operators.
