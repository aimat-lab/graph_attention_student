# Compiling GNN Models to WebAssembly for Browser Inference

Design document for a system that takes a trained PyTorch GNN model and compiles it into a WebAssembly module for client-side inference in the browser. The compiled module produces full model outputs — predictions, node/edge importance maps, and graph embeddings — enabling client-side rendering of explanation heatmaps without any server-side inference infrastructure.


> **See also:** [ADR-002](adr/002-onnx-export.md). An ONNX export now exists at
> `graph_attention_student/torch/export/`, covering the same browser use case with far
> less owned code. The two are complementary: ONNX is the *portability* path (Python,
> Node and the browser, on both the WASM and WebGPU backends), while this document
> describes the *speed* path. Both consume the same architecture description from
> `graph_attention_student/torch/export/architecture.py`, so they cannot disagree about
> what a model is. The explanation of why direct export was viable at all — MEGAN uses
> only `add` and `max` segment reductions, with no segment softmax anywhere — is in the
> ADR.

## Motivation and Performance Analysis

### Why browser inference

Running GNN inference directly in the user's browser eliminates three categories of cost:

1. **Infrastructure cost**: No GPU/CPU inference servers to provision, scale, or maintain. The user's device is the compute.
2. **Latency**: No network round-trip for predictions. For interactive use cases (draw a molecule, see the prediction + explanation), the response is instant.
3. **Privacy**: Molecular structures never leave the user's browser. This matters for proprietary compounds in pharmaceutical settings.

The key insight is that molecular GNNs operate on tiny graphs. A typical drug-like molecule has 10–50 nodes and 20–100 edges. The actual compute for a MEGAN forward pass on such a graph is ~2.2M MACs — a modern browser can execute this in microseconds via WebAssembly. The bottleneck in PyTorch is never the math; it's Python dispatch, tensor allocation, and PyG's message-passing plumbing. WebAssembly eliminates all of that overhead.

### Why this works

PyTorch Geometric on CPU spends most of its time on framework overhead, not compute — especially for small molecular graphs:

- Python interpreter dispatch between every operation
- PyTorch's C++ dispatcher + operator lookup (even with `torch.no_grad()`)
- A new tensor allocation for every intermediate result (gather, concat, linear, batchnorm, activation, scatter — each is a separate allocation + kernel)
- PyG's Python-level `MessagePassing` plumbing: `propagate()` → `message()` → `aggregate()` → `update()`, each a Python method call
- `scatter_add` / `scatter_max` on CPU: generic implementations that handle arbitrary dtypes, dims, and shapes

WebAssembly eliminates all of this. The compiled module is a single monolithic function with:
- Zero allocation in the hot path (pre-allocated scratch buffers)
- No dispatch overhead (all operations are inlined)
- No interpreter (wasm executes at near-native speed)
- Fused operations (linear + BN + activation in one loop)

### Expected performance (estimated, pending benchmarks)

| Scenario | PyTorch CPU | Wasm (single-threaded) | Wasm (Web Worker pool, 4 workers) |
|---|---|---|---|
| Single molecule | 1–5 ms | **0.1–0.5 ms** | 0.1–0.5 ms (no benefit) |
| Batch of 32 | 10–30 ms | **3–15 ms** | **1–5 ms** |
| Batch of 1000 | 200–800 ms | **100–500 ms** | **30–150 ms** |

**Important caveats on these estimates:**
- These are theoretical projections based on operation counts and expected overhead reduction. Actual performance must be benchmarked in target browsers before committing to production use.
- The batch-of-1000 numbers assume sequential per-molecule processing. They do NOT benefit from batched matmul the way PyTorch does, so the speedup ratio shrinks for large batches.
- Worker pool numbers assume negligible inter-worker communication overhead, which holds for large enough chunks but breaks down for very small batches.

**Wasm SIMD**: Wasm SIMD provides 128-bit vectors (equivalent to SSE2). The Rust compiler auto-vectorizes simple scalar loops (activation functions, element-wise ops) to wasm v128 instructions when compiled with `-C target-feature=+simd128`. However, the `matrixmultiply` crate's hand-tuned SIMD microkernels target x86 (SSE/AVX/FMA) and aarch64 (NEON) specifically — **on wasm32, it falls back to a generic scalar kernel**. LLVM may auto-vectorize parts of this generic kernel, but it will not match the hand-tuned native performance. For the small matrices in molecular GNNs (~30 rows), the matmul dispatch overhead matters more than SIMD width, so this fallback is acceptable. The expected slowdown vs native Rust is **2–4x**, which still represents a **5–20x speedup** over PyTorch CPU for single-molecule inference.

### Where this approach fits best

- **Interactive applications**: User draws or selects a molecule, gets instant prediction + explanation heatmap.
- **Screening tools**: Upload a set of SMILES/graphs, get batch predictions client-side. Hundreds to low thousands of molecules.
- **Educational/demo apps**: Show how the model works without requiring users to install Python or access a server.
- **Offline-capable apps**: Once the wasm module and weights are cached, inference works without network.

### Where this approach has limitations

- **Very large batches** (100K+ molecules): A server with native Rust + rayon will be faster. Wasm is single-threaded per instance; Web Workers help but have communication overhead.
- **Large graphs** (1000+ nodes): The matmul itself becomes the bottleneck and wasm's narrower SIMD hurts more. Molecular GNNs rarely hit this.
- **Model updates**: Changing the model requires recompilation and re-deployment of the wasm binary + weights. Not suitable for models that change hourly.
- **Mobile browsers**: Performance is good on desktop browsers but may be slower on mobile due to thermal throttling and weaker CPUs.

## Architecture Overview

### Compilation Pipeline (runs once, on developer's machine)

```
Python compile_gnn_wasm() call
    │
    ├── 0. Pre-flight check (wasm-pack, rustc wasm32 target)
    ├── 1. Introspect PyTorch model → extract architecture graph + weight tensors
    ├── 2. Apply compile-time optimizations (BN folding into linear layers)
    ├── 3. Serialize optimized weights to binary file (magic + version + CRC32)
    ├── 4. Generate Rust source from Jinja2 templates (wasm-bindgen interface)
    ├── 5. wasm-pack build --target web --release → pkg/ directory
    └── 6. Output: .wasm + .js glue + weights.bin → ready for CDN upload
```

### Browser Runtime

```
CDN (static files)
    ├── model_bg.wasm          (~100–300 KB gzipped; ndarray adds bulk)
    ├── model.js               (~5 KB, ES module glue)
    └── weights.bin            (~1–5 MB, cached by browser)

Browser (user's device)
    │
    ├── Main Thread
    │     ├── fetch() wasm + weights from CDN (cached after first load)
    │     ├── WebAssembly.instantiateStreaming() → wasm instance
    │     ├── Load weights into wasm memory
    │     ├── model.forward(graph) → prediction + explanations
    │     └── Render heatmap from importance values
    │
    └── Web Worker Pool (for batch processing)
          ├── Worker 1: own wasm instance + weights → process molecules 0..N/4
          ├── Worker 2: own wasm instance + weights → process molecules N/4..N/2
          ├── Worker 3: own wasm instance + weights → process molecules N/2..3N/4
          └── Worker 4: own wasm instance + weights → process molecules 3N/4..N
```

The system has two Rust components:

1. **`gnn-runtime`** — a reusable crate with optimized building blocks (fused linear+BN+activation, scatter ops, gather+concat, etc.). Uses the `matrixmultiply` crate for matmuls. Pure Rust, no system dependencies, compiles to wasm32 without modification.
2. **Generated model crate** — auto-generated Rust code that wires the runtime ops together for a specific model architecture, with `wasm-bindgen` bindings for the JavaScript interface.

## Component Details

### 1. Python Compiler (`compile_gnn_wasm()` function)

The entry point. Validates the environment (wasm-pack + wasm32 target), walks the PyTorch module tree, extracts architecture info and weights, generates Rust code, and compiles to wasm.

```python
from jinja2 import Template
import subprocess
import shutil
import struct
import numpy as np
import torch
import torch.nn as nn


class CompileError(Exception):
    """Raised when the WebAssembly compilation pipeline fails."""
    pass


def _check_wasm_toolchain():
    """
    Verify wasm-pack and rustc with wasm32-unknown-unknown target are installed.
    Raises CompileError with install instructions if not found.
    """
    if shutil.which("wasm-pack") is None:
        raise CompileError(
            "'wasm-pack' not found. Install it:\n"
            "  curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh\n"
            "Or: cargo install wasm-pack"
        )
    
    for cmd in ["cargo", "rustc"]:
        if shutil.which(cmd) is None:
            raise CompileError(
                f"'{cmd}' not found. Install the Rust toolchain: "
                "https://rustup.rs/ — then run: rustup default stable"
            )
    
    # Check that wasm32-unknown-unknown target is installed
    result = subprocess.run(
        ["rustup", "target", "list", "--installed"],
        capture_output=True, text=True,
    )
    if "wasm32-unknown-unknown" not in result.stdout:
        raise CompileError(
            "wasm32-unknown-unknown target not installed. Run:\n"
            "  rustup target add wasm32-unknown-unknown"
        )


def compile_gnn_wasm(
    model: nn.Module,
    output_dir: str,
    model_name: str = "megan_wasm",
    enable_simd: bool = True,
) -> str:
    """
    Compile a trained PyTorch GNN model into a WebAssembly module.
    
    Args:
        model: Trained PyTorch model in eval mode.
        output_dir: Directory where the Rust project and compiled artifacts are written.
        model_name: Name for the generated crate and wasm module.
        enable_simd: Enable wasm SIMD instructions (requires Chrome 91+, Firefox 89+,
                     Safari 16.4+). Disable for maximum compatibility.
        
    Returns:
        Path to the pkg/ directory containing the deployable artifacts.
        
    Raises:
        CompileError: If toolchain is missing or compilation fails.
    """
    # Step 0: Pre-flight check
    _check_wasm_toolchain()
    
    model.eval()
    
    # Step 0.5: Validate model is a supported architecture
    _validate_model(model)
    
    # Step 1: Extract architecture as a list of operation descriptors
    arch = extract_architecture(model)
    
    # Step 2: Fold BatchNorm into preceding linear layers (compile-time optimization)
    state = fold_batchnorms(model)
    
    # Step 2.5: Validate all BatchNorm layers were folded
    _validate_no_remaining_batchnorm(model, state)
    
    # Step 2.6: Sanity-check weights for NaN/Inf before serialization
    _validate_weights(state)
    
    # Step 3: Extract LayerNorm parameters (cannot be folded — runtime op)
    layer_norm_params = extract_layer_norms(model)
    state.update(layer_norm_params)
    
    # Step 4: Serialize weights with magic + version + checksum
    weights_path = f"{output_dir}/weights.bin"
    write_weights_bin(state, weights_path)
    
    # Step 5: Generate Rust source from templates
    rust_code = WASM_FORWARD_TEMPLATE.render(arch=arch, model_name=model_name)
    
    import os
    os.makedirs(f"{output_dir}/src", exist_ok=True)
    write_file(f"{output_dir}/src/lib.rs", rust_code)
    write_file(f"{output_dir}/Cargo.toml",
               WASM_CARGO_TEMPLATE.render(name=model_name))
    
    # Step 6: Build with wasm-pack
    env = os.environ.copy()
    rustflags = []
    if enable_simd:
        rustflags.append("-C target-feature=+simd128")
    rustflags.append("-C opt-level=3")
    env["RUSTFLAGS"] = " ".join(rustflags)
    
    result = subprocess.run(
        ["wasm-pack", "build", "--target", "web", "--release", "--no-typescript"],
        cwd=output_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CompileError(
            f"wasm-pack build failed:\n{result.stderr}\n"
            f"Generated source at: {output_dir}/src/lib.rs"
        )
    
    # Step 7: Copy weights.bin into pkg/ for deployment
    pkg_dir = f"{output_dir}/pkg"
    shutil.copy2(weights_path, f"{pkg_dir}/weights.bin")
    
    # Step 8: Generate JS convenience wrapper and Web Worker script
    write_file(f"{pkg_dir}/megan-loader.js",
               MEGAN_LOADER_TEMPLATE.render(model_name=model_name))
    write_file(f"{pkg_dir}/megan-worker.js",
               MEGAN_WORKER_TEMPLATE.render(model_name=model_name))
    write_file(f"{pkg_dir}/megan-pool.js",
               MEGAN_POOL_TEMPLATE.render())
    
    # Step 8.5: Generate content-hashed filenames for cache busting
    _add_content_hashes(pkg_dir)
    
    print(f"WebAssembly build complete. Deploy contents of: {pkg_dir}/")
    return pkg_dir


def _validate_model(model: nn.Module):
    """
    Validate that the model uses a supported architecture.
    Currently only MEGAN with V2 attention layers is supported.
    """
    layer_version = getattr(model, 'layer_type', None)
    if layer_version is not None and layer_version != 'v2':
        raise CompileError(
            f"Only V2 attention layers are supported for wasm compilation. "
            f"This model uses layer_type='{layer_version}'. "
            f"V1 and V3 support may be added in a future version."
        )
    
    # Check that BatchNorm running stats exist (model must have seen data)
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            if module.running_mean is None or module.running_var is None:
                raise CompileError(
                    f"BatchNorm1d '{name}' has no running statistics. "
                    f"The model must be trained (or at least run in train mode "
                    f"on data) before compilation."
                )


def _validate_no_remaining_batchnorm(
    model: nn.Module,
    state: dict[str, np.ndarray],
):
    """
    Assert that all BatchNorm1d layers have been folded into preceding
    linear layers. Raises CompileError if any are left unfolded.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            # Check if this BN was consumed by fold_batchnorms
            # (its preceding linear should have a ._fused marker)
            fused = any(k.endswith('._fused') for k in state)
            if not fused:
                raise CompileError(
                    f"BatchNorm1d '{name}' was not folded into a preceding "
                    f"linear layer. This may indicate an unsupported layer "
                    f"ordering (e.g., activation between Linear and BN). "
                    f"Check the model architecture."
                )


def _validate_weights(state: dict[str, np.ndarray]):
    """
    Sanity-check extracted weights for NaN or Inf values.
    These indicate a diverged or corrupted model.
    """
    for name, arr in state.items():
        if not isinstance(arr, np.ndarray):
            continue
        if np.any(np.isnan(arr)):
            raise CompileError(
                f"Weight tensor '{name}' contains NaN values. "
                f"The model may not be trained correctly."
            )
        if np.any(np.isinf(arr)):
            raise CompileError(
                f"Weight tensor '{name}' contains Inf values. "
                f"The model may not be trained correctly."
            )


def _add_content_hashes(pkg_dir: str):
    """
    Rename wasm and weights files with content hashes for cache busting.
    Write a manifest.json mapping original names to hashed names.
    """
    import hashlib
    import json
    import os
    
    manifest = {}
    for filename in os.listdir(pkg_dir):
        if filename.endswith(('.wasm', '.bin')):
            filepath = os.path.join(pkg_dir, filename)
            with open(filepath, 'rb') as f:
                content_hash = hashlib.sha256(f.read()).hexdigest()[:12]
            
            name, ext = os.path.splitext(filename)
            hashed_name = f"{name}.{content_hash}{ext}"
            hashed_path = os.path.join(pkg_dir, hashed_name)
            os.rename(filepath, hashed_path)
            manifest[filename] = hashed_name
    
    with open(os.path.join(pkg_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
```

### 2. Architecture Extraction

Walk the module tree and build a representation of the computation graph that the code generator can consume. This is identical regardless of compilation target (native or wasm).

```python
def extract_architecture(model: nn.Module) -> dict:
    """
    Extract a layer-by-layer description of the model architecture.
    
    Returns a dict with:
        "layers": list of operation descriptors, each a dict like:
            {"op": "linear", "name": "lay_embedd", "in_dim": 9, "out_dim": 64}
            {"op": "attention_layer", "name": "...", "variant": "v2",
             "in_dim": 64, "out_dim": 64, "edge_dim": 4, "hidden_dim": 128}
        "config": model-level configuration:
            {"attention_aggregation": "max", "num_channels": 2,
             "prediction_mode": "regression", "normalize_embedding": False,
             "output_norm": None}
    """
    arch = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            arch.append({
                "op": "linear",
                "name": name,
                "in_dim": module.in_features,
                "out_dim": module.out_features,
                "has_bias": module.bias is not None,
            })
        elif isinstance(module, nn.BatchNorm1d):
            arch.append({
                "op": "batchnorm",
                "name": name,
                "features": module.num_features,
                "eps": module.eps,
            })
        elif isinstance(module, nn.LayerNorm):
            arch.append({
                "op": "layer_norm",
                "name": name,
                "features": module.normalized_shape[0],
                "eps": module.eps,
            })
        elif isinstance(module, nn.LeakyReLU):
            arch.append({
                "op": "leaky_relu",
                "name": name,
                "negative_slope": module.negative_slope,
            })
        elif isinstance(module, nn.SiLU):
            arch.append({"op": "silu", "name": name})
        elif isinstance(module, nn.Tanh):
            arch.append({"op": "tanh", "name": name})
    
    arch_config = {
        "attention_aggregation": getattr(model, 'attention_aggregation', 'max'),
        "num_channels": getattr(model, 'num_channels', 2),
        "prediction_mode": getattr(model, 'prediction_mode', 'regression'),
        "normalize_embedding": getattr(model, 'normalize_embedding', False),
        "output_norm": getattr(model, 'output_norm', None),
    }
    
    return {"layers": arch, "config": arch_config}
```

### 3. BatchNorm Folding (Compile-Time Optimization)

At compile time, fold each BatchNorm's parameters into the preceding linear layer's weight and bias. This eliminates BatchNorm as a runtime operation entirely — important in wasm where every saved operation counts.

The math: given a linear layer `y = Wx + b` followed by BN with running mean μ, running var σ², learned scale γ, learned shift β, and epsilon ε:

```
BN(y) = γ * (y - μ) / sqrt(σ² + ε) + β
      = γ * (Wx + b - μ) / sqrt(σ² + ε) + β
      = (γ / sqrt(σ² + ε)) * Wx + (γ * (b - μ) / sqrt(σ² + ε) + β)
```

So the fused parameters are:
```
W_fused = W * (γ / sqrt(σ² + ε))    (scale each output row)
b_fused = (b - μ) * γ / sqrt(σ² + ε) + β
```

**Important**: LayerNorm CANNOT be folded because it normalizes across features at runtime (the mean and variance depend on the input). LayerNorm parameters (weight, bias) must be stored separately and applied as a runtime op.

```python
def fold_batchnorms(model: nn.Module) -> dict[str, np.ndarray]:
    """
    Walk the model and fold every BatchNorm1d into its preceding Linear layer.
    Returns a state dict with fused weights as numpy arrays.
    """
    state = {}
    modules = list(model.named_modules())
    
    for name, module in modules:
        if isinstance(module, nn.Sequential):
            children = list(module.children())
            i = 0
            while i < len(children):
                if (isinstance(children[i], nn.Linear)
                    and i + 1 < len(children)
                    and isinstance(children[i + 1], nn.BatchNorm1d)):
                    
                    linear = children[i]
                    bn = children[i + 1]
                    w_fused, b_fused = _fold_bn_into_linear(linear, bn)
                    
                    linear_name = f"{name}.{i}"
                    state[f"{linear_name}.weight"] = w_fused
                    state[f"{linear_name}.bias"] = b_fused
                    state[f"{linear_name}._fused"] = True
                    
                    i += 2  # skip the BN
                else:
                    if isinstance(children[i], nn.Linear):
                        linear = children[i]
                        linear_name = f"{name}.{i}"
                        state[f"{linear_name}.weight"] = (
                            linear.weight.detach().cpu().numpy()
                        )
                        if linear.bias is not None:
                            state[f"{linear_name}.bias"] = (
                                linear.bias.detach().cpu().numpy()
                            )
                    i += 1
    
    # Also store any standalone linear layers not in Sequential containers
    for name, module in modules:
        if isinstance(module, nn.Linear) and f"{name}.weight" not in state:
            state[f"{name}.weight"] = module.weight.detach().cpu().numpy()
            if module.bias is not None:
                state[f"{name}.bias"] = module.bias.detach().cpu().numpy()
    
    return state


def extract_layer_norms(model: nn.Module) -> dict[str, np.ndarray]:
    """Extract LayerNorm parameters (cannot be folded — must be applied at runtime)."""
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            state[f"{name}.weight"] = module.weight.detach().cpu().numpy()
            state[f"{name}.bias"] = module.bias.detach().cpu().numpy()
    return state


def _fold_bn_into_linear(linear: nn.Linear, bn: nn.BatchNorm1d):
    """Fold BatchNorm parameters into the preceding Linear layer."""
    w = linear.weight.data.clone()
    b = (linear.bias.data.clone() if linear.bias is not None
         else torch.zeros(w.shape[0]))
    
    mean = bn.running_mean.data
    var = bn.running_var.data
    gamma = bn.weight.data       # scale
    beta = bn.bias.data          # shift
    eps = bn.eps
    
    scale = gamma / torch.sqrt(var + eps)
    
    w_fused = w * scale.unsqueeze(1)
    b_fused = (b - mean) * scale + beta
    
    return w_fused.numpy(), b_fused.numpy()
```

### 4. Weight Serialization

Simple binary format with magic number, version, and CRC32 checksum for corruption detection. The format is target-agnostic — the same `weights.bin` file works for both native and wasm builds.

```python
import zlib

WEIGHTS_MAGIC = b"GNN\x01"
WEIGHTS_VERSION = 1


def write_weights_bin(state: dict[str, np.ndarray], path: str):
    """
    Serialize weight tensors to a simple binary format with integrity checks.
    
    Format:
        bytes[4]: magic ("GNN\x01")
        u32: format version
        u32: number of tensors
        For each tensor:
            u32: name length in bytes
            bytes: name (utf-8)
            u32: number of dimensions
            u32 * ndim: shape
            f32 * numel: data (little-endian, row-major)
        u32: CRC32 checksum of all preceding bytes
    """
    arrays = {k: v for k, v in state.items() if isinstance(v, np.ndarray)}
    
    buf = bytearray()
    buf += WEIGHTS_MAGIC
    buf += struct.pack('<I', WEIGHTS_VERSION)
    buf += struct.pack('<I', len(arrays))
    
    for name, arr in arrays.items():
        name_bytes = name.encode('utf-8')
        buf += struct.pack('<I', len(name_bytes))
        buf += name_bytes
        
        buf += struct.pack('<I', arr.ndim)
        for dim in arr.shape:
            buf += struct.pack('<I', dim)
        
        buf += arr.astype(np.float32).tobytes()
    
    # Append CRC32 checksum
    checksum = zlib.crc32(buf) & 0xFFFFFFFF
    buf += struct.pack('<I', checksum)
    
    with open(path, 'wb') as f:
        f.write(buf)
```

### 5. Rust Runtime Core Operations

The core operations crate. Uses `matrixmultiply` instead of full BLAS — this is critical for wasm because:

1. `matrixmultiply` is pure Rust with zero system dependencies. It compiles to `wasm32-unknown-unknown` without any special configuration.
2. Full BLAS (OpenBLAS, MKL) cannot be compiled to wasm. They require OS-level threading, system allocators, and platform-specific assembly.
3. For the small matrices in molecular GNNs (~30 rows × 128 cols), `matrixmultiply`'s low dispatch overhead outperforms BLAS anyway.

**Note on SIMD**: The Rust compiler auto-vectorizes simple scalar loops (activation functions, element-wise ops) to wasm SIMD with `-C target-feature=+simd128`. However, `matrixmultiply`'s hand-tuned SIMD microkernels only target x86 (SSE/AVX/FMA) and aarch64 (NEON). On `wasm32-unknown-unknown`, it falls back to a **generic scalar kernel**. LLVM may partially auto-vectorize this generic path, but it will not match native performance. For the small matrices in molecular GNNs this is acceptable — the dispatch overhead saved by using `matrixmultiply` over BLAS matters more than SIMD width.

**Note on binary size**: The `ndarray` crate adds significant generic machinery (shape checking, iterators, broadcasting) to the wasm binary. Realistic binary size with all dependencies is **200–400 KB gzipped**, not the ~50 KB achievable with flat `Vec<f32>` and manual stride math. This is a deliberate trade-off: ndarray provides correctness and readability at the cost of ~150 KB extra. If binary size becomes critical, ndarray can be replaced with manual indexing in a future optimization pass.

**Cargo.toml dependencies:**
```toml
[package]
name = "megan-wasm"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
ndarray = "0.16"
matrixmultiply = "0.3"
wasm-bindgen = "0.2"
js-sys = "0.3"
serde = { version = "1", features = ["derive"] }
serde-wasm-bindgen = "0.6"
console_error_panic_hook = "0.1"
crc32fast = "1"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
strip = "symbols"
panic = "abort"
```

Key differences from a native build:
- **No `rayon`**: wasm is single-threaded. Parallelism is handled at the JS level via Web Workers.
- **No `pyo3` / `numpy`**: replaced by `wasm-bindgen` and `js-sys` for the JavaScript interface.
- **`console_error_panic_hook`**: Converts Rust panics into readable browser console errors (development aid). Works with `panic = "abort"`.
- **`strip = "symbols"`**: Reduces wasm binary size by removing debug symbol names.
- **`panic = "abort"`**: Eliminates unwinding machinery, saving 10–30 KB in the wasm binary.
- **`crc32fast`**: Fast CRC32 implementation with lookup tables, replacing the naive bit-at-a-time algorithm.

#### Core Operations

```rust
// src/ops.rs

use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2, s};

/// Matrix multiply using the matrixmultiply crate.
/// output = input @ weight^T + bias (row-broadcasted)
/// All buffers are pre-allocated — zero heap allocations.
#[inline]
fn matmul_add_bias(
    input: &ArrayView2<f32>,   // (N, in_dim)
    weight: &ArrayView2<f32>,  // (out_dim, in_dim)
    bias: &Array1<f32>,        // (out_dim,)
    output: &mut ArrayViewMut2<f32>, // (N, out_dim)
) {
    let n = input.nrows();
    let k = input.ncols();     // in_dim
    let m = weight.nrows();    // out_dim
    
    // matrixmultiply computes C = alpha * A * B + beta * C
    // We need output = input @ weight^T
    // With row-major layout, we pass strides directly.
    //
    // Safety: strides must be non-negative (negative strides from reversed
    // slices would cause UB). All views come from pre-allocated scratch
    // buffers with standard row-major layout, so this holds.
    debug_assert!(input.strides().iter().all(|&s| s >= 0));
    debug_assert!(weight.strides().iter().all(|&s| s >= 0));
    debug_assert!(output.strides().iter().all(|&s| s >= 0));
    unsafe {
        matrixmultiply::sgemm(
            n, m, k,
            1.0,
            input.as_ptr(),
            input.strides()[0] as isize,
            input.strides()[1] as isize,
            weight.as_ptr(),
            weight.strides()[1] as isize,  // transposed
            weight.strides()[0] as isize,
            0.0,
            output.as_mut_ptr(),
            output.strides()[0] as isize,
            output.strides()[1] as isize,
        );
    }
    
    // Add bias (fused into same cache pass as the matmul output)
    for mut row in output.rows_mut() {
        row += &bias.view();
    }
}

/// Fused linear + leaky_relu in one pass.
/// Weight and bias already have BN folded in (done at compile time in Python).
/// Output buffer is pre-allocated — zero allocations in the hot path.
///
/// IMPORTANT: The embedding layer (lay_embedd) is a plain linear with NO activation.
/// Use linear() for embedding, not this function.
#[inline]
pub fn linear_leaky_relu(
    input: &ArrayView2<f32>,
    weight: &ArrayView2<f32>,
    bias: &Array1<f32>,
    negative_slope: f32,
    output: &mut ArrayViewMut2<f32>,
) {
    matmul_add_bias(input, weight, bias, output);
    output.mapv_inplace(|v| if v >= 0.0 { v } else { v * negative_slope });
}

/// Linear without activation (for embedding and final layers).
#[inline]
pub fn linear(
    input: &ArrayView2<f32>,
    weight: &ArrayView2<f32>,
    bias: &Array1<f32>,
    output: &mut ArrayViewMut2<f32>,
) {
    matmul_add_bias(input, weight, bias, output);
}

/// Linear + SiLU activation: x * sigmoid(x)
#[inline]
pub fn linear_silu(
    input: &ArrayView2<f32>,
    weight: &ArrayView2<f32>,
    bias: &Array1<f32>,
    output: &mut ArrayViewMut2<f32>,
) {
    matmul_add_bias(input, weight, bias, output);
    output.mapv_inplace(|v| v * (1.0 / (1.0 + (-v).exp())));
}

/// LayerNorm: normalize across features, then scale + shift.
/// Cannot be folded into linear — mean/variance depend on the input at runtime.
#[inline]
pub fn layer_norm(
    input: &mut ArrayViewMut2<f32>,  // (N, D) — modified in place
    weight: &Array1<f32>,            // (D,) learned scale
    bias: &Array1<f32>,              // (D,) learned shift
    eps: f32,
) {
    let d = input.ncols() as f32;
    for mut row in input.rows_mut() {
        let mean: f32 = row.iter().sum::<f32>() / d;
        let var: f32 = row.iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f32>() / d;
        let inv_std = 1.0 / (var + eps).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = (*v - mean) * inv_std * weight[i] + bias[i];
        }
    }
}

/// L2-normalize each row to unit length. Used when normalize_embedding=True.
#[inline]
pub fn l2_normalize(input: &mut ArrayViewMut2<f32>) {
    for mut row in input.rows_mut() {
        let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if norm > 1e-12 {
            row.mapv_inplace(|v| v / norm);
        }
    }
}

/// Compute neighbor sum: for each node, sum features of all neighbors.
/// neighbor_sum[i] = sum of node_features[src[e]] for all edges where dst[e] == i.
///
/// Precondition: all values in src and dst must be < num_nodes.
/// Validated at the forward() entry point; debug_assert here for safety.
pub fn neighbor_sum(
    node_features: &ArrayView2<f32>,  // (V, F)
    src: &[u32],                      // (E,) source node per edge
    dst: &[u32],                      // (E,) destination node per edge
    num_nodes: usize,
    output: &mut ArrayViewMut2<f32>,  // (V, F) — pre-allocated
) {
    output.slice_mut(s![..num_nodes, ..]).fill(0.0);
    for e in 0..src.len() {
        let s = src[e] as usize;
        let d = dst[e] as usize;
        debug_assert!(s < num_nodes && d < num_nodes,
            "edge {e}: src={s}, dst={d} out of bounds (num_nodes={num_nodes})");
        for f in 0..node_features.ncols() {
            output[[d, f]] += node_features[[s, f]];
        }
    }
}

/// Fused gather source + destination node features, neighbor sums,
/// and concatenate with edge features.
/// Output row = [x_src, x_dst, x_neighbor_dst, edge_attr]
pub fn gather_concat_with_neighbors(
    node_features: &ArrayView2<f32>,      // (V, F_node)
    neighbor_features: &ArrayView2<f32>,  // (V, F_node)
    edge_features: &ArrayView2<f32>,      // (E, F_edge)
    src: &[u32],
    dst: &[u32],
    output: &mut ArrayViewMut2<f32>,      // (E, 3*F_node + F_edge)
) {
    let f = node_features.ncols();
    for e in 0..src.len() {
        let s = src[e] as usize;
        let d = dst[e] as usize;
        let mut row = output.row_mut(e);
        row.slice_mut(s![..f]).assign(&node_features.row(s));
        row.slice_mut(s![f..2*f]).assign(&node_features.row(d));
        row.slice_mut(s![2*f..3*f]).assign(&neighbor_features.row(d));
        row.slice_mut(s![3*f..]).assign(&edge_features.row(e));
    }
}

/// Sum-aggregate values from edges to nodes.
pub fn scatter_sum(
    edge_values: &ArrayView2<f32>,   // (E, D)
    dst: &[u32],                     // (E,) destination node for each edge
    num_nodes: usize,
    output: &mut ArrayViewMut2<f32>, // (V, D)
) {
    output.slice_mut(s![..num_nodes, ..]).fill(0.0);
    for e in 0..dst.len() {
        let d = dst[e] as usize;
        for f in 0..edge_values.ncols() {
            output[[d, f]] += edge_values[[e, f]];
        }
    }
}

/// Scatter max: for each node, find the max value across its incoming edges.
pub fn scatter_max(
    edge_values: &ArrayView2<f32>,   // (E, K)
    index: &[u32],                   // (E,)
    num_nodes: usize,
    output: &mut ArrayViewMut2<f32>, // (V, K)
) {
    output.slice_mut(s![..num_nodes, ..]).fill(f32::NEG_INFINITY);
    for e in 0..index.len() {
        let n = index[e] as usize;
        for k in 0..edge_values.ncols() {
            let cur = output[[n, k]];
            let val = edge_values[[e, k]];
            if val > cur {
                output[[n, k]] = val;
            }
        }
    }
    // Replace -inf with 0 for nodes with no incoming edges
    for n in 0..num_nodes {
        for k in 0..output.ncols() {
            if output[[n, k]] == f32::NEG_INFINITY {
                output[[n, k]] = 0.0;
            }
        }
    }
}

/// Element-wise sigmoid in-place: 1 / (1 + exp(-x))
#[inline]
pub fn sigmoid_inplace(arr: &mut ArrayViewMut2<f32>) {
    arr.mapv_inplace(|v| 1.0 / (1.0 + (-v).exp()));
}

/// Element-wise tanh in-place
#[inline]
pub fn tanh_inplace(arr: &mut ArrayViewMut2<f32>) {
    arr.mapv_inplace(|v| v.tanh());
}

/// Replace inf values with 0 (matches PyTorch's workaround in megan.py:622).
#[inline]
pub fn replace_inf_with_zero(arr: &mut ArrayViewMut2<f32>) {
    arr.mapv_inplace(|v| if v.is_infinite() { 0.0 } else { v });
}
```

### 6. Pre-Allocated Scratch Buffers

Allocate all intermediate buffers once at initialization, reuse them across calls. This is the single biggest source of speedup for small graphs — it eliminates heap allocation from the hot path entirely. In wasm, where the allocator is simpler than the system allocator, avoiding allocation is even more important.

All ops take sliced views (`s![..num_edges, ..]`) of oversized buffers so that `fill(0.0)` only touches rows actually needed.

**Important**: Wasm linear memory can grow but never shrink (`memory.grow` is irreversible). A single outlier graph (e.g., a polymer with 10K nodes) would permanently inflate memory for the lifetime of the tab. The scratch buffers enforce a hard cap to prevent this and to protect against malicious inputs that could OOM the browser tab.

```rust
// src/scratch.rs

use ndarray::Array2;

/// Hard caps to prevent wasm memory exhaustion from outlier/malicious inputs.
/// Molecular graphs exceeding these limits are rejected at the forward() entry point.
const MAX_SUPPORTED_NODES: usize = 1024;
const MAX_SUPPORTED_EDGES: usize = 4096;

/// Pre-allocated buffers for a single-molecule forward pass.
/// In wasm (single-threaded), only one Scratch is needed.
/// For Web Worker parallelism, each worker has its own wasm instance
/// and therefore its own Scratch.
pub struct Scratch {
    // Per-layer message passing buffers
    pub edge_concat: Array2<f32>,      // (max_edges, concat_dim)
    pub msg_hidden: Array2<f32>,       // (max_edges, hidden_dim)
    pub msg_out: Array2<f32>,          // (max_edges, out_dim)
    pub attn_hidden: Array2<f32>,      // (max_edges, hidden_dim)
    pub attn_scores: Array2<f32>,      // (max_edges, 1)
    pub attn_sigmoid: Array2<f32>,     // (max_edges, 1)
    pub weighted_msg: Array2<f32>,     // (max_edges, out_dim)
    pub node_aggregated: Array2<f32>,  // (max_nodes, out_dim)
    pub neighbor_sum_buf: Array2<f32>, // (max_nodes, in_dim) for V2 layers
    pub transform_input: Array2<f32>,  // (max_nodes, out_dim + in_dim)
    pub transform_hidden: Array2<f32>, // (max_nodes, hidden_dim)
    pub node_out: Array2<f32>,         // (max_nodes, out_dim)
    
    // Per-channel node embeddings
    pub channel_embs: Vec<Array2<f32>>,  // num_channels × (max_nodes, out_dim)
    
    // Attention logits storage: [layer][channel] -> (max_edges, 1)
    pub alpha_storage: Vec<Vec<Array2<f32>>>,
    
    // Importance computation buffers
    pub edge_importance: Array2<f32>,      // (max_edges, num_channels)
    pub node_imp_src: Array2<f32>,         // (max_nodes, num_channels)
    pub node_imp_dst: Array2<f32>,         // (max_nodes, num_channels)
    pub node_importance: Array2<f32>,      // (max_nodes, num_channels)
    pub node_importance_norm: Array2<f32>, // (max_nodes, num_channels)
    pub edge_importance_norm: Array2<f32>, // (max_edges, num_channels)
    
    // Graph-level buffers
    pub graph_embedding: Array2<f32>,  // (1, embed_dim)
    pub projection_buf: Array2<f32>,   // (1, proj_dim)
    
    // Embedding buffer
    pub node_emb: Array2<f32>,         // (max_nodes, embed_dim)
}

impl Scratch {
    pub fn new(max_nodes: usize, max_edges: usize, dims: &ModelDims) -> Self {
        Scratch {
            edge_concat: Array2::zeros((max_edges, dims.concat_dim)),
            msg_hidden: Array2::zeros((max_edges, dims.hidden_dim)),
            msg_out: Array2::zeros((max_edges, dims.out_dim)),
            attn_hidden: Array2::zeros((max_edges, dims.hidden_dim)),
            attn_scores: Array2::zeros((max_edges, 1)),
            attn_sigmoid: Array2::zeros((max_edges, 1)),
            weighted_msg: Array2::zeros((max_edges, dims.out_dim)),
            node_aggregated: Array2::zeros((max_nodes, dims.out_dim)),
            neighbor_sum_buf: Array2::zeros((max_nodes, dims.max_in_dim)),
            transform_input: Array2::zeros((max_nodes, dims.out_dim + dims.max_in_dim)),
            transform_hidden: Array2::zeros((max_nodes, dims.hidden_dim)),
            node_out: Array2::zeros((max_nodes, dims.out_dim)),
            channel_embs: (0..dims.num_channels)
                .map(|_| Array2::zeros((max_nodes, dims.out_dim)))
                .collect(),
            alpha_storage: (0..dims.num_layers)
                .map(|_| (0..dims.num_channels)
                    .map(|_| Array2::zeros((max_edges, 1)))
                    .collect())
                .collect(),
            edge_importance: Array2::zeros((max_edges, dims.num_channels)),
            node_imp_src: Array2::zeros((max_nodes, dims.num_channels)),
            node_imp_dst: Array2::zeros((max_nodes, dims.num_channels)),
            node_importance: Array2::zeros((max_nodes, dims.num_channels)),
            node_importance_norm: Array2::zeros((max_nodes, dims.num_channels)),
            edge_importance_norm: Array2::zeros((max_edges, dims.num_channels)),
            graph_embedding: Array2::zeros((1, dims.embed_dim * dims.num_channels)),
            projection_buf: Array2::zeros((1, dims.proj_dim)),
            node_emb: Array2::zeros((max_nodes, dims.embed_dim)),
        }
    }
    
    /// Grow buffers if current molecule exceeds allocated capacity.
    /// Only reallocates buffers that are too small. Never shrinks.
    /// Returns Err if the molecule exceeds hard limits.
    pub fn ensure_capacity(
        &mut self,
        num_nodes: usize,
        num_edges: usize,
        dims: &ModelDims,
    ) -> Result<(), String> {
        if num_nodes > MAX_SUPPORTED_NODES {
            return Err(format!(
                "Graph has {num_nodes} nodes, exceeding limit of {MAX_SUPPORTED_NODES}. \
                 Molecular GNNs are designed for small graphs."
            ));
        }
        if num_edges > MAX_SUPPORTED_EDGES {
            return Err(format!(
                "Graph has {num_edges} edges, exceeding limit of {MAX_SUPPORTED_EDGES}."
            ));
        }
        if num_edges > self.edge_concat.nrows() {
            let new_cap = (num_edges * 3) / 2;  // grow by 1.5x
            self.edge_concat = Array2::zeros((new_cap, dims.concat_dim));
            self.msg_hidden = Array2::zeros((new_cap, dims.hidden_dim));
            self.msg_out = Array2::zeros((new_cap, dims.out_dim));
            self.attn_hidden = Array2::zeros((new_cap, dims.hidden_dim));
            self.attn_scores = Array2::zeros((new_cap, 1));
            self.attn_sigmoid = Array2::zeros((new_cap, 1));
            self.weighted_msg = Array2::zeros((new_cap, dims.out_dim));
            self.edge_importance = Array2::zeros((new_cap, dims.num_channels));
            self.edge_importance_norm = Array2::zeros((new_cap, dims.num_channels));
            for layer_alphas in &mut self.alpha_storage {
                for alpha in layer_alphas.iter_mut() {
                    *alpha = Array2::zeros((new_cap, 1));
                }
            }
        }
        if num_nodes > self.node_aggregated.nrows() {
            let new_cap = (num_nodes * 3) / 2;
            self.node_aggregated = Array2::zeros((new_cap, dims.out_dim));
            self.neighbor_sum_buf = Array2::zeros((new_cap, dims.max_in_dim));
            self.transform_input = Array2::zeros((
                new_cap, dims.out_dim + dims.max_in_dim,
            ));
            self.transform_hidden = Array2::zeros((new_cap, dims.hidden_dim));
            self.node_out = Array2::zeros((new_cap, dims.out_dim));
            self.node_emb = Array2::zeros((new_cap, dims.embed_dim));
            for emb in &mut self.channel_embs {
                *emb = Array2::zeros((new_cap, dims.out_dim));
            }
            self.node_imp_src = Array2::zeros((new_cap, dims.num_channels));
            self.node_imp_dst = Array2::zeros((new_cap, dims.num_channels));
            self.node_importance = Array2::zeros((new_cap, dims.num_channels));
            self.node_importance_norm = Array2::zeros((new_cap, dims.num_channels));
        }
        Ok(())
    }
}
```

### 7. Generated Model Code (wasm-bindgen)

The Jinja2 template produces a complete Rust module for a specific model, with `wasm-bindgen` bindings instead of PyO3. Here's what the generated code looks like for a MEGAN model (GraphAttentionLayerV2, 3 layers, 64 units, hidden_dim=128, 2 channels).

The generated source includes a preamble comment with the model config and `// [template: ...]` markers at key sections so that Rust compiler errors can be traced back to the template that produced them.

```rust
// Auto-generated by gnn_compiler (wasm target) — do not edit manually
// Model: MEGAN (v2, layers=3, units=[64,64,64], hidden=128, channels=2)
// Attention aggregation: max
// Prediction mode: regression
// Source checkpoint: /path/to/model.ckpt

mod ops;
mod scratch;

use ops::*;
use scratch::Scratch;
use ndarray::prelude::*;
use wasm_bindgen::prelude::*;
use js_sys::Float32Array;

// [template: constants]
const NUM_CHANNELS: usize = 2;
const NUM_LAYERS: usize = 3;
const EMBED_DIM: usize = 64;
const NODE_DIM: usize = 9;
const EDGE_DIM: usize = 4;

/// Initialize panic hook for readable browser console errors.
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// All model weights, loaded from weights.bin at initialization.
struct Weights {
    // Embedding: node_dim -> units[0] (plain linear, NO activation)
    embed_w: Array2<f32>,  // (64, 9)
    embed_b: Array1<f32>,  // (64,)
    
    // Per channel, per layer: attention layer weights
    channels: Vec<ChannelWeights>,
    
    // Projection layers per channel
    proj: Vec<Vec<ProjectionLayerParams>>,
    
    // Final dense prediction head
    final_layers: Vec<DenseLayerParams>,
    
    // Regression reference bias (or 0.0 for classification)
    regression_reference: f32,
    
    // Model config
    attention_aggregation: AttentionAggregation,
    normalize_embedding: bool,
    output_norm: Option<f32>,
}

enum AttentionAggregation { Sum, Mean, Max, Min }

struct ChannelWeights {
    layers: Vec<AttentionLayerV2Weights>,
}

struct AttentionLayerV2Weights {
    // Message MLP: (3*in_dim + edge_dim) -> hidden -> out
    // BN already folded into linear weights at compile time
    msg_w0: Array2<f32>,
    msg_b0: Array1<f32>,
    msg_w1: Array2<f32>,   // no activation after this layer
    msg_b1: Array1<f32>,
    
    // Attention MLP: (3*in_dim + edge_dim) -> hidden -> hidden -> 1
    attn_w0: Array2<f32>,
    attn_b0: Array1<f32>,
    attn_w1: Array2<f32>,
    attn_b1: Array1<f32>,
    attn_w2: Array2<f32>,  // (1, hidden)
    attn_b2: Array1<f32>,
    
    // Transform MLP: (out + in) -> hidden -> out
    trans_w0: Array2<f32>,
    trans_b0: Array1<f32>,
    trans_w1: Array2<f32>,
    trans_b1: Array1<f32>,
}

struct ProjectionLayerParams {
    weight: Array2<f32>,
    bias: Array1<f32>,
    ln_weight: Array1<f32>,
    ln_bias: Array1<f32>,
    ln_eps: f32,
}

struct DenseLayerParams {
    weight: Array2<f32>,
    bias: Array1<f32>,
    ln_weight: Option<Array1<f32>>,
    ln_bias: Option<Array1<f32>>,
    is_last: bool,
}


// ──────────────────────────────────────────────────────────────
// Weight loading from ArrayBuffer (fetched from CDN)
// ──────────────────────────────────────────────────────────────

const WEIGHTS_MAGIC: &[u8; 4] = b"GNN\x01";
const WEIGHTS_VERSION: u32 = 1;

impl Weights {
    /// Load weights from a byte buffer (received from JavaScript as &[u8]).
    /// Validates magic, version, and CRC32 checksum.
    /// All slice accesses are bounds-checked — a truncated file returns Err,
    /// never panics.
    fn from_bytes(data: &[u8]) -> Result<Self, String> {
        // Minimum file size: magic(4) + version(4) + count(4) + crc(4) = 16
        if data.len() < 16 {
            return Err(format!(
                "Weight file too small ({} bytes, minimum 16)", data.len()
            ));
        }
        
        // Verify CRC32 (last 4 bytes) using crc32fast crate
        let stored_crc = u32::from_le_bytes(
            data[data.len()-4..].try_into().map_err(|_| "truncated")?
        );
        let computed_crc = crc32fast::hash(&data[..data.len()-4]);
        if stored_crc != computed_crc {
            return Err(format!(
                "Weight file corrupted: CRC32 mismatch \
                 (stored={stored_crc:#x}, computed={computed_crc:#x})"
            ));
        }
        
        let mut offset = 0;
        
        // Magic
        let magic = read_bytes(data, &mut offset, 4)?;
        if magic != WEIGHTS_MAGIC {
            return Err("Not a GNN weights file (bad magic)".into());
        }
        
        // Version
        let version = read_u32(data, &mut offset)?;
        if version != WEIGHTS_VERSION {
            return Err(format!("Unsupported weight format version: {version}"));
        }
        
        // Number of tensors
        let count = read_u32(data, &mut offset)? as usize;
        
        // Read all tensors into a HashMap
        let mut tensors = std::collections::HashMap::new();
        for tensor_idx in 0..count {
            let name_len = read_u32(data, &mut offset)? as usize;
            let name_bytes = read_bytes(data, &mut offset, name_len)?;
            let name = std::str::from_utf8(name_bytes)
                .map_err(|_| format!(
                    "invalid utf-8 in tensor name at index {tensor_idx}"
                ))?
                .to_string();
            
            let ndim = read_u32(data, &mut offset)? as usize;
            let mut shape = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                shape.push(read_u32(data, &mut offset)? as usize);
            }
            
            let numel: usize = shape.iter().product();
            let byte_len = numel * 4;
            let float_bytes = read_bytes(data, &mut offset, byte_len)?;
            let arr = Array1::from(
                float_bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<f32>>()
            );
            
            match ndim {
                1 => { tensors.insert(name, TensorData::D1(arr)); }
                2 => {
                    tensors.insert(
                        name.clone(),
                        TensorData::D2(arr.into_shape_with_order(
                            (shape[0], shape[1])
                        ).map_err(|e| format!(
                            "reshape failed for tensor '{name}': {e}"
                        ))?),
                    );
                }
                _ => return Err(format!(
                    "unsupported ndim={ndim} for tensor '{name}'"
                )),
            }
        }
        
        // Build Weights struct from tensor map
        // (template-generated code that pulls tensors by name)
        Self::from_tensor_map(tensors)
    }
}

/// Read a u32 from the buffer with bounds checking.
fn read_u32(data: &[u8], offset: &mut usize) -> Result<u32, String> {
    let end = *offset + 4;
    let bytes = data.get(*offset..end)
        .ok_or_else(|| format!(
            "unexpected end of weights file at offset {}", *offset
        ))?;
    *offset = end;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

/// Read N bytes from the buffer with bounds checking.
fn read_bytes<'a>(
    data: &'a [u8], offset: &mut usize, len: usize,
) -> Result<&'a [u8], String> {
    let end = *offset + len;
    let bytes = data.get(*offset..end)
        .ok_or_else(|| format!(
            "unexpected end of weights file at offset {} \
             (wanted {} bytes, {} available)",
            *offset, len, data.len() - *offset
        ))?;
    *offset = end;
    Ok(bytes)
}


// ──────────────────────────────────────────────────────────────
// Forward pass for a SINGLE molecule
// ──────────────────────────────────────────────────────────────

/// Result of a single molecule forward pass.
/// All arrays are owned (not views into scratch).
struct SingleForwardResult {
    prediction: Array1<f32>,
    node_importance: Array2<f32>,      // (V, K)
    node_importance_norm: Array2<f32>, // (V, K)
    edge_importance: Array2<f32>,      // (E, K)
    edge_importance_norm: Array2<f32>, // (E, K)
    graph_embedding: Vec<Array1<f32>>, // K × (embed_dim,)
}

// [template: forward_single]
fn forward_single(
    weights: &Weights,
    scratch: &mut Scratch,
    node_attr: &ArrayView2<f32>,  // (V, 9)
    edge_attr: &ArrayView2<f32>,  // (E, 4)
    src: &[u32],                  // (E,)
    dst: &[u32],                  // (E,)
    predict_only: bool,           // skip importance computation
) -> Result<SingleForwardResult, String> {
    let num_nodes = node_attr.nrows();
    let num_edges = src.len();
    
    scratch.ensure_capacity(num_nodes, num_edges, &DIMS)
        .map_err(|e| e)?;  // propagate size-cap errors
    
    // --- Validate edge indices are within bounds ---
    for e in 0..num_edges {
        if src[e] as usize >= num_nodes || dst[e] as usize >= num_nodes {
            return Err(format!(
                "edge {e}: src={} dst={} out of bounds (num_nodes={num_nodes})",
                src[e], dst[e]
            ));
        }
    }
    
    // --- Replace inf with 0 (matches megan.py:622) ---
    let mut node_input = node_attr.to_owned();
    replace_inf_with_zero(&mut node_input.view_mut());
    let mut edge_input = edge_attr.to_owned();
    replace_inf_with_zero(&mut edge_input.view_mut());
    
    // --- Embedding (plain linear, NO activation) ---
    // [template: embedding]
    {
        let mut emb_view = scratch.node_emb.slice_mut(s![..num_nodes, ..]);
        linear(
            &node_input.view(), &weights.embed_w.view(), &weights.embed_b,
            &mut emb_view,
        );
    }
    
    // Initialize per-channel embeddings from shared embedding
    for ch in 0..NUM_CHANNELS {
        scratch.channel_embs[ch]
            .slice_mut(s![..num_nodes, ..])
            .assign(&scratch.node_emb.slice(s![..num_nodes, ..]));
    }
    
    // --- Encoder layers ---
    for layer_idx in 0..NUM_LAYERS {
        for ch in 0..NUM_CHANNELS {
            // [template: encoder_layer, ch={ch}, layer={layer_idx}]
            let layer_w = &weights.channels[ch].layers[layer_idx];
            
            let x = scratch.channel_embs[ch]
                .slice(s![..num_nodes, ..]).to_owned();
            
            // 1. Neighbor sum
            neighbor_sum(
                &x.view(), src, dst, num_nodes,
                &mut scratch.neighbor_sum_buf.slice_mut(s![..num_nodes, ..]),
            );
            
            // 2. Gather + concat: [x_src, x_dst, x_neighbor_dst, edge_attr]
            gather_concat_with_neighbors(
                &x.view(),
                &scratch.neighbor_sum_buf.slice(s![..num_nodes, ..]),
                edge_attr, src, dst,
                &mut scratch.edge_concat.slice_mut(s![..num_edges, ..]),
            );
            
            // 3. Message MLP (BN pre-folded)
            linear_leaky_relu(
                &scratch.edge_concat.slice(s![..num_edges, ..]),
                &layer_w.msg_w0.view(), &layer_w.msg_b0, 0.01,
                &mut scratch.msg_hidden.slice_mut(s![..num_edges, ..]),
            );
            linear(
                &scratch.msg_hidden.slice(s![..num_edges, ..]),
                &layer_w.msg_w1.view(), &layer_w.msg_b1,
                &mut scratch.msg_out.slice_mut(s![..num_edges, ..]),
            );
            
            // 4. Attention MLP
            linear_leaky_relu(
                &scratch.edge_concat.slice(s![..num_edges, ..]),
                &layer_w.attn_w0.view(), &layer_w.attn_b0, 0.01,
                &mut scratch.attn_hidden.slice_mut(s![..num_edges, ..]),
            );
            linear_leaky_relu(
                &scratch.attn_hidden.slice(s![..num_edges, ..]),
                &layer_w.attn_w1.view(), &layer_w.attn_b1, 0.01,
                &mut scratch.attn_hidden.slice_mut(s![..num_edges, ..]),
            );
            linear(
                &scratch.attn_hidden.slice(s![..num_edges, ..]),
                &layer_w.attn_w2.view(), &layer_w.attn_b2,
                &mut scratch.attn_scores.slice_mut(s![..num_edges, ..]),
            );
            
            // 5. Sigmoid attention * message
            scratch.attn_sigmoid.slice_mut(s![..num_edges, ..])
                .assign(&scratch.attn_scores.slice(s![..num_edges, ..]));
            sigmoid_inplace(
                &mut scratch.attn_sigmoid.slice_mut(s![..num_edges, ..]),
            );
            
            for e in 0..num_edges {
                let alpha = scratch.attn_sigmoid[[e, 0]];
                for d in 0..scratch.msg_out.ncols() {
                    scratch.weighted_msg[[e, d]] =
                        alpha * scratch.msg_out[[e, d]];
                }
            }
            
            // 6. Scatter sum: aggregate to destination nodes
            scatter_sum(
                &scratch.weighted_msg.slice(s![..num_edges, ..]),
                dst, num_nodes,
                &mut scratch.node_aggregated.slice_mut(s![..num_nodes, ..]),
            );
            
            // 7. Transform MLP: [aggregated, x] -> hidden -> out
            let in_dim = x.ncols();
            let out_dim = scratch.node_aggregated.ncols();
            for n in 0..num_nodes {
                for d in 0..out_dim {
                    scratch.transform_input[[n, d]] =
                        scratch.node_aggregated[[n, d]];
                }
                for d in 0..in_dim {
                    scratch.transform_input[[n, out_dim + d]] = x[[n, d]];
                }
            }
            
            linear_leaky_relu(
                &scratch.transform_input.slice(s![..num_nodes, ..]),
                &layer_w.trans_w0.view(), &layer_w.trans_b0, 0.01,
                &mut scratch.transform_hidden.slice_mut(s![..num_nodes, ..]),
            );
            linear_leaky_relu(
                &scratch.transform_hidden.slice(s![..num_nodes, ..]),
                &layer_w.trans_w1.view(), &layer_w.trans_b1, 0.01,
                &mut scratch.node_out.slice_mut(s![..num_nodes, ..]),
            );
            
            // Residual connection (layers.py:428)
            for n in 0..num_nodes {
                for d in 0..out_dim {
                    scratch.channel_embs[ch][[n, d]] =
                        scratch.node_out[[n, d]] + x[[n, d]];
                }
            }
            
            // Store attention logits for importance computation
            // (only needed when computing explanations)
            if !predict_only {
                scratch.alpha_storage[layer_idx][ch]
                    .slice_mut(s![..num_edges, ..])
                    .assign(&scratch.attn_scores.slice(s![..num_edges, ..]));
            }
        }
    }
    
    // --- Importance computation (skipped in predict-only mode) ---
    // [template: importance]
    if predict_only {
        // Skip importance, normalization, and projection — jump straight
        // to graph embedding with uniform weights
        return self.forward_predict_only_tail(
            weights, scratch, num_nodes, num_edges,
        );
    }
    for k in 0..NUM_CHANNELS {
        for e in 0..num_edges {
            let agg = match weights.attention_aggregation {
                AttentionAggregation::Max => {
                    let mut val = f32::NEG_INFINITY;
                    for l in 0..NUM_LAYERS {
                        let sig = 1.0 / (1.0 + (
                            -scratch.alpha_storage[l][k][[e, 0]]
                        ).exp());
                        val = val.max(sig);
                    }
                    val
                }
                AttentionAggregation::Min => {
                    let mut val = f32::INFINITY;
                    for l in 0..NUM_LAYERS {
                        let sig = 1.0 / (1.0 + (
                            -scratch.alpha_storage[l][k][[e, 0]]
                        ).exp());
                        val = val.min(sig);
                    }
                    val
                }
                AttentionAggregation::Sum => {
                    let mut val = 0.0f32;
                    for l in 0..NUM_LAYERS {
                        val += 1.0 / (1.0 + (
                            -scratch.alpha_storage[l][k][[e, 0]]
                        ).exp());
                    }
                    val
                }
                AttentionAggregation::Mean => {
                    let mut val = 0.0f32;
                    for l in 0..NUM_LAYERS {
                        val += 1.0 / (1.0 + (
                            -scratch.alpha_storage[l][k][[e, 0]]
                        ).exp());
                    }
                    val / NUM_LAYERS as f32
                }
            };
            scratch.edge_importance[[e, k]] = agg;
        }
    }
    
    // Node importance: mean of max-pool from src and dst endpoints
    scatter_max(
        &scratch.edge_importance.slice(s![..num_edges, ..]),
        src, num_nodes,
        &mut scratch.node_imp_src.slice_mut(s![..num_nodes, ..]),
    );
    scatter_max(
        &scratch.edge_importance.slice(s![..num_edges, ..]),
        dst, num_nodes,
        &mut scratch.node_imp_dst.slice_mut(s![..num_nodes, ..]),
    );
    
    for n in 0..num_nodes {
        for k in 0..NUM_CHANNELS {
            scratch.node_importance[[n, k]] = 0.5 * (
                scratch.node_imp_src[[n, k]]
                + scratch.node_imp_dst[[n, k]]
            );
        }
    }
    
    // Normalize node importance per graph
    let mut max_val = 0.0f32;
    for n in 0..num_nodes {
        for k in 0..NUM_CHANNELS {
            max_val = max_val.max(scratch.node_importance[[n, k]]);
        }
    }
    let divisor = if max_val < 0.01 { 1e9 } else { max_val };
    
    for n in 0..num_nodes {
        for k in 0..NUM_CHANNELS {
            let normed = scratch.node_importance[[n, k]] / divisor;
            scratch.node_importance_norm[[n, k]] =
                if normed < 0.1 { normed * 0.01 } else { normed };
        }
    }
    
    // Normalize edge importance per graph
    let mut edge_max = 0.0f32;
    for e in 0..num_edges {
        for k in 0..NUM_CHANNELS {
            edge_max = edge_max.max(scratch.edge_importance[[e, k]]);
        }
    }
    let edge_divisor = if edge_max < 0.01 { 1e9 } else { edge_max };
    
    for e in 0..num_edges {
        for k in 0..NUM_CHANNELS {
            let normed = scratch.edge_importance[[e, k]] / edge_divisor;
            scratch.edge_importance_norm[[e, k]] =
                if normed < 0.1 { normed * 0.01 } else { normed };
        }
    }
    
    // --- Graph embedding: importance-weighted sum pooling per channel ---
    // [template: graph_embedding]
    let embed_dim = EMBED_DIM;
    let mut graph_embedding_channels: Vec<Array1<f32>> =
        Vec::with_capacity(NUM_CHANNELS);
    
    for k in 0..NUM_CHANNELS {
        let mut ge = Array1::zeros(embed_dim);
        for n in 0..num_nodes {
            let weight = scratch.node_importance_norm[[n, k]];
            for d in 0..embed_dim {
                ge[d] += weight * scratch.channel_embs[k][[n, d]];
            }
        }
        
        // Projection layers: Linear -> SiLU -> LayerNorm (per layer)
        // [template: projection, ch={k}]
        if !weights.proj[k].is_empty() {
            let mut proj_input = ge.insert_axis(ndarray::Axis(0));
            
            for proj_params in &weights.proj[k] {
                let mut proj_out =
                    Array2::zeros((1, proj_params.weight.nrows()));
                linear_silu(
                    &proj_input.view(),
                    &proj_params.weight.view(),
                    &proj_params.bias,
                    &mut proj_out.view_mut(),
                );
                layer_norm(
                    &mut proj_out.view_mut(),
                    &proj_params.ln_weight,
                    &proj_params.ln_bias,
                    proj_params.ln_eps,
                );
                proj_input = proj_out;
            }
            
            ge = proj_input.row(0).to_owned();
        }
        
        // Optional L2 normalization
        if weights.normalize_embedding {
            let norm: f32 = ge.iter().map(|&v| v * v).sum::<f32>().sqrt();
            if norm > 1e-12 {
                ge.mapv_inplace(|v| v / norm);
            }
        }
        
        graph_embedding_channels.push(ge);
    }
    
    // --- Final prediction MLP ---
    // [template: final_mlp]
    let total_dim: usize =
        graph_embedding_channels.iter().map(|e| e.len()).sum();
    let mut concat = Array2::zeros((1, total_dim));
    let mut offset = 0;
    for ge in &graph_embedding_channels {
        for d in 0..ge.len() {
            concat[[0, offset + d]] = ge[d];
        }
        offset += ge.len();
    }
    
    let mut x = concat;
    for params in &weights.final_layers {
        let mut out = Array2::zeros((1, params.weight.nrows()));
        if params.is_last {
            linear(
                &x.view(), &params.weight.view(), &params.bias,
                &mut out.view_mut(),
            );
            out.mapv_inplace(|v| v + weights.regression_reference);
            
            if let Some(norm_val) = weights.output_norm {
                l2_normalize(&mut out.view_mut());
                out.mapv_inplace(|v| v * norm_val);
            }
        } else {
            linear_silu(
                &x.view(), &params.weight.view(), &params.bias,
                &mut out.view_mut(),
            );
            if let (Some(w), Some(b)) = (&params.ln_weight, &params.ln_bias) {
                layer_norm(&mut out.view_mut(), w, b, 1e-5);
            }
        }
        x = out;
    }
    
    // --- Collect results ---
    Ok(SingleForwardResult {
        prediction: x.row(0).to_owned(),
        node_importance: scratch.node_importance
            .slice(s![..num_nodes, ..]).to_owned(),
        node_importance_norm: scratch.node_importance_norm
            .slice(s![..num_nodes, ..]).to_owned(),
        edge_importance: scratch.edge_importance
            .slice(s![..num_edges, ..]).to_owned(),
        edge_importance_norm: scratch.edge_importance_norm
            .slice(s![..num_edges, ..]).to_owned(),
        graph_embedding: graph_embedding_channels,
    })
}


// ──────────────────────────────────────────────────────────────
// wasm-bindgen interface
// ──────────────────────────────────────────────────────────────
// [template: wasm_bindings]

#[wasm_bindgen]
pub struct WasmMegan {
    weights: Weights,
    scratch: Scratch,
}

#[wasm_bindgen]
impl WasmMegan {
    /// Create a new model instance from a weights buffer.
    /// The buffer is the contents of weights.bin, fetched from CDN.
    #[wasm_bindgen(constructor)]
    pub fn new(weights_data: &[u8]) -> Result<WasmMegan, JsError> {
        let weights = Weights::from_bytes(weights_data)
            .map_err(|e| JsError::new(&e))?;
        let scratch = Scratch::new(128, 256, &DIMS);
        Ok(WasmMegan { weights, scratch })
    }
    
    /// Full forward pass for a single molecule.
    ///
    /// Arguments (all row-major, flattened):
    ///   node_attr: Float32Array of length (num_nodes × node_dim)
    ///   edge_attr: Float32Array of length (num_edges × edge_dim)
    ///   edge_indices: Uint32Array of length (num_edges × 2),
    ///                 layout: [src_0, dst_0, src_1, dst_1, ...]
    ///   num_nodes: number of nodes
    ///   num_edges: number of edges
    ///
    /// Returns a JS object with Float32Array fields:
    ///   prediction, node_importance, node_importance_norm,
    ///   edge_importance, edge_importance_norm, graph_embedding
    pub fn forward(
        &mut self,
        node_attr: &[f32],
        edge_attr: &[f32],
        edge_indices: &[u32],
        num_nodes: usize,
        num_edges: usize,
    ) -> Result<JsValue, JsError> {
        // Reshape flat arrays into 2D views
        let node_2d = ArrayView2::from_shape(
            (num_nodes, NODE_DIM), node_attr
        ).map_err(|e| JsError::new(&format!("node_attr shape: {e}")))?;
        
        let edge_2d = ArrayView2::from_shape(
            (num_edges, EDGE_DIM), edge_attr
        ).map_err(|e| JsError::new(&format!("edge_attr shape: {e}")))?;
        
        // Split edge_indices into src and dst arrays
        let mut src = Vec::with_capacity(num_edges);
        let mut dst = Vec::with_capacity(num_edges);
        for i in 0..num_edges {
            src.push(edge_indices[i * 2]);
            dst.push(edge_indices[i * 2 + 1]);
        }
        
        let result = forward_single(
            &self.weights, &mut self.scratch,
            &node_2d, &edge_2d, &src, &dst,
            false,  // predict_only = false → compute explanations
        ).map_err(|e| JsError::new(&e))?;
        
        // Build JS return object with Float32Array fields
        let obj = js_sys::Object::new();
        
        set_float32_field(&obj, "prediction",
            result.prediction.as_slice().unwrap())?;
        set_float32_field(&obj, "node_importance",
            result.node_importance.as_slice().unwrap())?;
        set_float32_field(&obj, "node_importance_norm",
            result.node_importance_norm.as_slice().unwrap())?;
        set_float32_field(&obj, "edge_importance",
            result.edge_importance.as_slice().unwrap())?;
        set_float32_field(&obj, "edge_importance_norm",
            result.edge_importance_norm.as_slice().unwrap())?;
        
        // Flatten graph_embedding: concat all channel embeddings
        let ge_flat: Vec<f32> = result.graph_embedding.iter()
            .flat_map(|ch| ch.iter().cloned())
            .collect();
        set_float32_field(&obj, "graph_embedding", &ge_flat)?;
        
        // Include shape metadata for client-side reshaping
        js_sys::Reflect::set(
            &obj, &"num_nodes".into(),
            &JsValue::from(num_nodes as u32),
        ).unwrap();
        js_sys::Reflect::set(
            &obj, &"num_edges".into(),
            &JsValue::from(num_edges as u32),
        ).unwrap();
        js_sys::Reflect::set(
            &obj, &"num_channels".into(),
            &JsValue::from(NUM_CHANNELS as u32),
        ).unwrap();
        js_sys::Reflect::set(
            &obj, &"embed_dim".into(),
            &JsValue::from(EMBED_DIM as u32),
        ).unwrap();
        
        Ok(obj.into())
    }
    
    /// Predict only (no explanations). Skips importance computation
    /// and projection layers for ~2x faster inference.
    pub fn predict(
        &mut self,
        node_attr: &[f32],
        edge_attr: &[f32],
        edge_indices: &[u32],
        num_nodes: usize,
        num_edges: usize,
    ) -> Result<Float32Array, JsError> {
        let node_2d = ArrayView2::from_shape(
            (num_nodes, NODE_DIM), node_attr
        ).map_err(|e| JsError::new(&format!("node_attr shape: {e}")))?;
        
        let edge_2d = ArrayView2::from_shape(
            (num_edges, EDGE_DIM), edge_attr
        ).map_err(|e| JsError::new(&format!("edge_attr shape: {e}")))?;
        
        let mut src = Vec::with_capacity(num_edges);
        let mut dst = Vec::with_capacity(num_edges);
        for i in 0..num_edges {
            src.push(edge_indices[i * 2]);
            dst.push(edge_indices[i * 2 + 1]);
        }
        
        let result = forward_single(
            &self.weights, &mut self.scratch,
            &node_2d, &edge_2d, &src, &dst,
            true,  // predict_only = true → skip explanations
        ).map_err(|e| JsError::new(&e))?;
        
        Ok(Float32Array::from(result.prediction.as_slice().unwrap()))
    }
    
    /// Process a batch of molecules sequentially.
    ///
    /// For better performance on large batches, prefer the Web Worker pool
    /// (MeganWorkerPool in megan-pool.js) which parallelizes across workers.
    ///
    /// Note: each graph in the JS array crosses the JS/wasm boundary
    /// individually. For batches over ~50 graphs, the Worker pool approach
    /// is recommended as it amortizes initialization overhead.
    ///
    /// Accepts a JS Array of graph objects, each with:
    ///   { node_attr: Float32Array, edge_attr: Float32Array,
    ///     edge_indices: Uint32Array, num_nodes: number, num_edges: number }
    ///
    /// Returns a JS Array of result objects.
    pub fn forward_batch(
        &mut self,
        graphs: &JsValue,
    ) -> Result<JsValue, JsError> {
        let arr = js_sys::Array::from(graphs);
        let results = js_sys::Array::new();
        
        for i in 0..arr.length() {
            let graph = arr.get(i);
            
            let node_attr = get_f32_field(&graph, "node_attr")?;
            let edge_attr = get_f32_field(&graph, "edge_attr")?;
            let edge_indices = get_u32_field(&graph, "edge_indices")?;
            let num_nodes = get_usize_field(&graph, "num_nodes")?;
            let num_edges = get_usize_field(&graph, "num_edges")?;
            
            let result = self.forward(
                &node_attr, &edge_attr, &edge_indices,
                num_nodes, num_edges,
            )?;
            results.push(&result);
        }
        
        Ok(results.into())
    }
    
    /// Release the wasm memory held by this model instance.
    /// Call this when the model is no longer needed to avoid leaking
    /// wasm linear memory (which cannot be reclaimed by the GC).
    /// After calling dispose(), this instance must not be used again.
    pub fn dispose(self) {
        // wasm-bindgen generates a .free() method on all exported structs.
        // Dropping `self` here releases the Weights and Scratch allocations
        // in wasm linear memory. Note: wasm linear memory pages themselves
        // are not returned to the OS, but the allocator can reuse them.
        drop(self);
    }
    
    /// Returns model metadata as a JS object.
    #[wasm_bindgen(getter)]
    pub fn metadata(&self) -> JsValue {
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(
            &obj, &"num_channels".into(),
            &JsValue::from(NUM_CHANNELS as u32),
        ).unwrap();
        js_sys::Reflect::set(
            &obj, &"num_layers".into(),
            &JsValue::from(NUM_LAYERS as u32),
        ).unwrap();
        js_sys::Reflect::set(
            &obj, &"embed_dim".into(),
            &JsValue::from(EMBED_DIM as u32),
        ).unwrap();
        js_sys::Reflect::set(
            &obj, &"node_dim".into(),
            &JsValue::from(NODE_DIM as u32),
        ).unwrap();
        js_sys::Reflect::set(
            &obj, &"edge_dim".into(),
            &JsValue::from(EDGE_DIM as u32),
        ).unwrap();
        obj.into()
    }
}

/// Helper: set a Float32Array field on a JS object.
fn set_float32_field(
    obj: &js_sys::Object,
    key: &str,
    data: &[f32],
) -> Result<(), JsError> {
    let arr = Float32Array::from(data);
    js_sys::Reflect::set(obj, &key.into(), &arr.into())
        .map_err(|_| JsError::new(&format!("failed to set field '{key}'")))?;
    Ok(())
}

/// Helper: read a Float32Array field from a JS object.
fn get_f32_field(obj: &JsValue, key: &str) -> Result<Vec<f32>, JsError> {
    let val = js_sys::Reflect::get(obj, &key.into())
        .map_err(|_| JsError::new(&format!("missing field '{key}'")))?;
    let arr: Float32Array = val.unchecked_into();
    Ok(arr.to_vec())
}

/// Helper: read a Uint32Array field from a JS object.
fn get_u32_field(obj: &JsValue, key: &str) -> Result<Vec<u32>, JsError> {
    let val = js_sys::Reflect::get(obj, &key.into())
        .map_err(|_| JsError::new(&format!("missing field '{key}'")))?;
    let arr: js_sys::Uint32Array = val.unchecked_into();
    Ok(arr.to_vec())
}

/// Helper: read a numeric field from a JS object.
fn get_usize_field(obj: &JsValue, key: &str) -> Result<usize, JsError> {
    let val = js_sys::Reflect::get(obj, &key.into())
        .map_err(|_| JsError::new(&format!("missing field '{key}'")))?;
    val.as_f64()
        .map(|v| v as usize)
        .ok_or_else(|| JsError::new(&format!("field '{key}' is not a number")))
}
```

### 8. JavaScript API

The compiled wasm module is consumed via a convenience loader that handles fetching, initialization, and provides a clean API. All types use vanilla JS — no framework dependencies.

#### TypeScript Definitions

```typescript
// megan.d.ts — TypeScript type definitions for the MEGAN wasm module.
// These describe the JS-side API, not the wasm-bindgen generated glue.

/**
 * Input graph representation. Framework-agnostic.
 * All arrays are flat (row-major) typed arrays.
 */
interface GraphInput {
    /** Node feature matrix, row-major. Length = num_nodes × node_dim. */
    node_attr: Float32Array;
    /** Edge feature matrix, row-major. Length = num_edges × edge_dim. */
    edge_attr: Float32Array;
    /**
     * Edge index pairs as [src_0, dst_0, src_1, dst_1, ...].
     * Length = num_edges × 2.
     */
    edge_indices: Uint32Array;
    /** Number of nodes in the graph. */
    num_nodes: number;
    /** Number of edges in the graph. */
    num_edges: number;
}

/**
 * Full forward pass result. All arrays are flat (row-major).
 * Use num_nodes, num_edges, num_channels, embed_dim to reshape.
 */
interface ForwardResult {
    /** Model prediction. Length = num_targets. */
    prediction: Float32Array;
    /**
     * Node importance per channel, row-major.
     * Shape: (num_nodes × num_channels).
     * node_importance[n * num_channels + k] = importance of node n in channel k.
     */
    node_importance: Float32Array;
    /**
     * Normalized node importance (same layout as node_importance).
     * Values in [0, 1] per graph, suitable for heatmap coloring.
     */
    node_importance_norm: Float32Array;
    /**
     * Edge importance per channel, row-major.
     * Shape: (num_edges × num_channels).
     */
    edge_importance: Float32Array;
    /** Normalized edge importance (same layout). */
    edge_importance_norm: Float32Array;
    /**
     * Graph-level embedding per channel, concatenated.
     * Shape: (num_channels × embed_dim).
     * graph_embedding[k * embed_dim + d] = dimension d of channel k's embedding.
     */
    graph_embedding: Float32Array;

    /** Shape metadata for client-side reshaping. */
    num_nodes: number;
    num_edges: number;
    num_channels: number;
    embed_dim: number;
}

/**
 * Model metadata (read-only properties baked in at compile time).
 */
interface ModelMetadata {
    num_channels: number;
    num_layers: number;
    embed_dim: number;
    node_dim: number;
    edge_dim: number;
}

/**
 * Progress callback for weight loading.
 * `total` is null when Content-Length header is absent (e.g., chunked transfer).
 */
type ProgressCallback = (loaded: number, total: number | null) => void;

/**
 * Configuration for MeganModel.load().
 */
interface LoadConfig {
    /** Base URL where model files are hosted (e.g., 'https://cdn.example.com/megan/'). */
    baseUrl: string;
    /** Optional progress callback for weight download. */
    onProgress?: ProgressCallback;
    /** Override wasm filename (default: 'model_bg.wasm'). */
    wasmFilename?: string;
    /** Override weights filename (default: 'weights.bin'). */
    weightsFilename?: string;
    /**
     * Optional: pass an already-imported wasm-bindgen module.
     * This avoids dynamic import() calls that can break bundlers (Vite, webpack).
     * Usage: import * as wasmModule from './megan_wasm.js';
     *        MeganModel.load({ wasmModule, baseUrl: '...' });
     */
    wasmModule?: { default: (input?: any) => Promise<any>; WasmMegan: any };
}
```

#### Convenience Loader

```javascript
// megan-loader.js — Convenience wrapper for loading and using the MEGAN wasm module.
// Vanilla JS, no framework dependencies.

/**
 * High-level interface for the compiled MEGAN model.
 * Handles wasm initialization, weight loading, and provides
 * a clean API for single and batch inference.
 *
 * Usage:
 *   const model = await MeganModel.load({
 *     baseUrl: 'https://cdn.example.com/megan/',
 *   });
 *   const result = model.forward(graph);
 */
export class MeganModel {
    /** @type {WasmMegan} */
    #wasm;

    constructor(wasmInstance) {
        this.#wasm = wasmInstance;
    }

    /**
     * Load model from CDN.
     * @param {LoadConfig} config
     * @returns {Promise<MeganModel>}
     */
    static async load(config) {
        const {
            baseUrl,
            onProgress,
            wasmFilename = 'megan_wasm_bg.wasm',
            weightsFilename = 'weights.bin',
            wasmModule: providedModule,
        } = config;

        const base = baseUrl.endsWith('/') ? baseUrl : baseUrl + '/';

        // Load wasm module and weights in parallel.
        // If the caller provided a pre-imported wasm module (recommended
        // for bundler compatibility), use it directly.
        const [wasmModule, weightsBuffer] = await Promise.all([
            providedModule
                ? Promise.resolve(providedModule)
                : MeganModel.#loadWasmModule(base + wasmFilename),
            MeganModel.#loadWeights(base + weightsFilename, onProgress),
        ]);

        // Initialize wasm — import the generated JS glue
        const { default: init, WasmMegan } = wasmModule;
        await init();

        // Create model instance with weights
        const instance = new WasmMegan(new Uint8Array(weightsBuffer));
        return new MeganModel(instance);
    }

    /**
     * Full forward pass for a single graph.
     * Returns prediction + all explanation outputs.
     * @param {GraphInput} graph
     * @returns {ForwardResult}
     */
    forward(graph) {
        return this.#wasm.forward(
            graph.node_attr,
            graph.edge_attr,
            graph.edge_indices,
            graph.num_nodes,
            graph.num_edges,
        );
    }

    /**
     * Predict only (no explanations).
     * @param {GraphInput} graph
     * @returns {Float32Array}
     */
    predict(graph) {
        return this.#wasm.predict(
            graph.node_attr,
            graph.edge_attr,
            graph.edge_indices,
            graph.num_nodes,
            graph.num_edges,
        );
    }

    /**
     * Batch forward pass (sequential, single-threaded).
     * For parallel batch processing, use MeganWorkerPool instead.
     * @param {GraphInput[]} graphs
     * @returns {ForwardResult[]}
     */
    forwardBatch(graphs) {
        return this.#wasm.forward_batch(graphs);
    }

    /**
     * Batch predict (sequential, single-threaded).
     * @param {GraphInput[]} graphs
     * @returns {Float32Array[]}
     */
    predictBatch(graphs) {
        return graphs.map(g => this.predict(g));
    }

    /**
     * Model metadata (num_channels, num_layers, embed_dim, etc.).
     * @returns {ModelMetadata}
     */
    get metadata() {
        return this.#wasm.metadata;
    }

    /**
     * Release wasm memory held by this model. Call when the model is no
     * longer needed to avoid leaking wasm linear memory.
     * After calling dispose(), this instance must not be used again.
     */
    dispose() {
        if (this.#wasm) {
            this.#wasm.free();  // wasm-bindgen generated cleanup
            this.#wasm = null;
        }
    }

    /**
     * Support for TC39 Explicit Resource Management proposal.
     * Enables: using model = await MeganModel.load(config);
     */
    [Symbol.dispose]() {
        this.dispose();
    }

    // ── Private helpers ──

    static async #loadWasmModule(url) {
        // Dynamic import of the wasm-pack generated JS glue.
        // The glue module exports `default` (init function) and
        // the wasm-bindgen classes.
        return await import(url.replace('_bg.wasm', '.js'));
    }

    static async #loadWeights(url, onProgress) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(
                `Failed to fetch weights: ${response.status} ${response.statusText}`
            );
        }

        if (!onProgress) {
            return await response.arrayBuffer();
        }

        // Stream with progress tracking
        const contentLengthHeader = response.headers.get('Content-Length');
        const total = contentLengthHeader
            ? parseInt(contentLengthHeader, 10)
            : null;  // null when Content-Length is absent (chunked transfer)
        const reader = response.body.getReader();
        const chunks = [];
        let loaded = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            loaded += value.length;
            onProgress(loaded, total);
        }

        // Merge chunks into a single ArrayBuffer
        const merged = new Uint8Array(loaded);
        let offset = 0;
        for (const chunk of chunks) {
            merged.set(chunk, offset);
            offset += chunk.length;
        }
        return merged.buffer;
    }
}
```

#### Usage Example (Vanilla JS)

```html
<script type="module">
import { MeganModel } from './megan-loader.js';

// Load model from CDN
const model = await MeganModel.load({
    baseUrl: 'https://cdn.example.com/megan/',
    onProgress: (loaded, total) => {
        const pct = total ? `${(loaded/total*100).toFixed(0)}%` : `${loaded} bytes`;
        console.log(`Loading weights: ${pct}`);
    },
});

console.log('Model loaded:', model.metadata);
// → { num_channels: 2, num_layers: 3, embed_dim: 64, node_dim: 9, edge_dim: 4 }

// Define a graph (e.g., a simple 3-node molecule)
const graph = {
    node_attr: new Float32Array([
        // node 0: 9 features
        1, 0, 0, 0, 0, 0, 0, 0, 0,
        // node 1: 9 features
        0, 1, 0, 0, 0, 0, 0, 0, 0,
        // node 2: 9 features
        0, 0, 1, 0, 0, 0, 0, 0, 0,
    ]),
    edge_attr: new Float32Array([
        // edge 0→1: 4 features
        1, 0, 0, 0,
        // edge 1→0: 4 features
        1, 0, 0, 0,
        // edge 1→2: 4 features
        0, 1, 0, 0,
        // edge 2→1: 4 features
        0, 1, 0, 0,
    ]),
    edge_indices: new Uint32Array([
        0, 1,  // edge 0: src=0, dst=1
        1, 0,  // edge 1: src=1, dst=0
        1, 2,  // edge 2: src=1, dst=2
        2, 1,  // edge 3: src=2, dst=1
    ]),
    num_nodes: 3,
    num_edges: 4,
};

// Run full forward pass
const result = model.forward(graph);

console.log('Prediction:', result.prediction);
// → Float32Array [value]

// Access importance for heatmap rendering
const { num_nodes, num_channels } = result;
for (let n = 0; n < num_nodes; n++) {
    for (let k = 0; k < num_channels; k++) {
        const imp = result.node_importance_norm[n * num_channels + k];
        console.log(`Node ${n}, channel ${k}: importance = ${imp.toFixed(3)}`);
    }
}
</script>
```

### 9. Batch Processing with Web Workers

Wasm is single-threaded — there is no equivalent to rayon's thread pool. For batch inference (tens to thousands of molecules), Web Workers provide parallelism: each worker loads its own wasm instance with its own linear memory and scratch buffers, processes a chunk of molecules independently, and returns results to the main thread.

#### Worker Script

```javascript
// megan-worker.js — Web Worker script for parallel batch inference.
// Each worker has its own wasm instance, weights, and scratch buffers.

let model = null;

self.onmessage = async (event) => {
    const { type, requestId, data } = event.data;

    switch (type) {
        case 'init': {
            try {
                // Dynamic import of the wasm-pack generated module.
                // Requires module workers: new Worker(url, { type: 'module' })
                const { default: init, WasmMegan } = await import(data.moduleUrl);
                await init(data.wasmUrl);

                // Load weights into this worker's wasm instance
                model = new WasmMegan(new Uint8Array(data.weightsBuffer));

                self.postMessage({ type: 'ready' });
            } catch (err) {
                self.postMessage({
                    type: 'error',
                    requestId,
                    error: err.message,
                });
            }
            break;
        }

        case 'forward_batch': {
            try {
                const { graphs, startIndex } = data;
                const results = [];

                for (const graph of graphs) {
                    const result = model.forward(
                        graph.node_attr,
                        graph.edge_attr,
                        graph.edge_indices,
                        graph.num_nodes,
                        graph.num_edges,
                    );
                    results.push(result);
                }

                self.postMessage({
                    type: 'batch_result',
                    requestId,
                    startIndex,
                    results,
                });
            } catch (err) {
                self.postMessage({
                    type: 'error',
                    requestId,
                    error: err.message,
                });
            }
            break;
        }

        case 'predict_batch': {
            try {
                const { graphs, startIndex } = data;
                const predictions = [];

                for (const graph of graphs) {
                    const pred = model.predict(
                        graph.node_attr,
                        graph.edge_attr,
                        graph.edge_indices,
                        graph.num_nodes,
                        graph.num_edges,
                    );
                    predictions.push(pred);
                }

                self.postMessage({
                    type: 'predict_result',
                    requestId,
                    startIndex,
                    predictions,
                });
            } catch (err) {
                self.postMessage({
                    type: 'error',
                    requestId,
                    error: err.message,
                });
            }
            break;
        }
    }
};
```

#### Worker Pool Coordinator

```javascript
// megan-pool.js — Web Worker pool for parallel batch inference.
// Distributes molecules across workers and collects results.

export class MeganWorkerPool {
    /** @type {Worker[]} */
    #workers = [];
    /** @type {number} */
    #numWorkers;
    /** @type {boolean} */
    #ready = false;
    /** @type {number} */
    #nextRequestId = 0;
    /**
     * Pending request handlers, keyed by requestId.
     * @type {Map<number, { resolve: Function, reject: Function, timer: number }>}
     */
    #pending = new Map();
    /** @type {number} Timeout per batch dispatch in ms (default: 30s) */
    #timeoutMs;

    /**
     * @param {number} numWorkers — Number of Web Workers.
     *   Default: navigator.hardwareConcurrency || 4
     * @param {{ timeoutMs?: number }} options
     */
    constructor(numWorkers, { timeoutMs = 30000 } = {}) {
        this.#numWorkers = numWorkers
            || (typeof navigator !== 'undefined'
                ? navigator.hardwareConcurrency || 4
                : 4);
        this.#timeoutMs = timeoutMs;
    }

    /**
     * Initialize the worker pool: spawn workers, load wasm + weights
     * in each one.
     *
     * @param {string} baseUrl — CDN base URL where model files are hosted.
     * @returns {Promise<void>}
     */
    async init(baseUrl) {
        const base = baseUrl.endsWith('/') ? baseUrl : baseUrl + '/';
        const moduleUrl = base + 'megan_wasm.js';
        const wasmUrl = base + 'megan_wasm_bg.wasm';
        const weightsUrl = base + 'weights.bin';

        // Fetch weights once on the main thread
        const weightsResponse = await fetch(weightsUrl);
        if (!weightsResponse.ok) {
            throw new Error(
                `Failed to fetch weights: ${weightsResponse.status}`
            );
        }
        const weightsBuffer = await weightsResponse.arrayBuffer();

        // Spawn workers and initialize each with wasm + weights
        const readyPromises = [];

        for (let i = 0; i < this.#numWorkers; i++) {
            const worker = new Worker(
                new URL('./megan-worker.js', import.meta.url),
                { type: 'module' },
            );
            this.#workers.push(worker);

            // Persistent message handler — routes responses by requestId.
            // This prevents the race condition where overlapping dispatches
            // overwrite each other's onmessage handlers.
            worker.onmessage = (event) => {
                const { type, requestId } = event.data;
                if (type === 'ready') {
                    // Handled by the init promise below
                    return;
                }
                const pending = this.#pending.get(requestId);
                if (!pending) return;  // stale or duplicate response
                clearTimeout(pending.timer);
                this.#pending.delete(requestId);

                if (type === 'error') {
                    pending.reject(new Error(event.data.error));
                } else {
                    const key = type === 'batch_result'
                        ? 'results' : 'predictions';
                    pending.resolve(event.data[key]);
                }
            };

            // Handle worker crashes (OOM, uncaught exceptions, etc.)
            worker.onerror = (err) => {
                console.error(`MEGAN worker ${i} crashed:`, err);
                // Reject all pending requests for this worker
                for (const [id, pending] of this.#pending) {
                    clearTimeout(pending.timer);
                    pending.reject(new Error(
                        `Worker ${i} crashed: ${err.message || 'unknown error'}`
                    ));
                    this.#pending.delete(id);
                }
            };

            const readyPromise = new Promise((resolve, reject) => {
                const origHandler = worker.onmessage;
                worker.onmessage = (event) => {
                    if (event.data.type === 'ready') {
                        worker.onmessage = origHandler;  // restore persistent handler
                        resolve();
                    } else if (event.data.type === 'error') {
                        reject(new Error(event.data.error));
                    }
                };
            });

            // Each worker gets its own copy of the weights buffer.
            // For typical MEGAN models (~1-5 MB), this is acceptable.
            worker.postMessage({
                type: 'init',
                data: {
                    moduleUrl,
                    wasmUrl,
                    weightsBuffer: weightsBuffer.slice(0),
                },
            });

            readyPromises.push(readyPromise);
        }

        await Promise.all(readyPromises);
        this.#ready = true;
    }

    /**
     * Parallel forward pass on a batch of graphs.
     * Splits the batch across workers, collects and reorders results.
     *
     * @param {GraphInput[]} graphs
     * @returns {Promise<ForwardResult[]>}
     */
    async forwardBatch(graphs) {
        if (!this.#ready) {
            throw new Error('Worker pool not initialized. Call init() first.');
        }
        return this.#dispatchBatch('forward_batch', 'batch_result', graphs);
    }

    /**
     * Parallel predict on a batch of graphs.
     * @param {GraphInput[]} graphs
     * @returns {Promise<Float32Array[]>}
     */
    async predictBatch(graphs) {
        if (!this.#ready) {
            throw new Error('Worker pool not initialized. Call init() first.');
        }
        return this.#dispatchBatch('predict_batch', 'predict_result', graphs);
    }

    /**
     * Shut down all workers and reject all pending requests.
     */
    terminate() {
        for (const [id, pending] of this.#pending) {
            clearTimeout(pending.timer);
            pending.reject(new Error('Worker pool terminated'));
        }
        this.#pending.clear();
        for (const worker of this.#workers) {
            worker.terminate();
        }
        this.#workers = [];
        this.#ready = false;
    }

    // ── Private ──

    async #dispatchBatch(requestType, responseType, graphs) {
        const n = this.#numWorkers;
        const chunkSize = Math.ceil(graphs.length / n);

        const promises = this.#workers.map((worker, i) => {
            const start = i * chunkSize;
            const end = Math.min(start + chunkSize, graphs.length);
            const chunk = graphs.slice(start, end);

            if (chunk.length === 0) {
                return Promise.resolve([]);
            }

            const requestId = this.#nextRequestId++;

            return new Promise((resolve, reject) => {
                // Timeout guard — prevents hanging if worker is stuck
                const timer = setTimeout(() => {
                    this.#pending.delete(requestId);
                    reject(new Error(
                        `Worker ${i} timed out after ${this.#timeoutMs}ms`
                    ));
                }, this.#timeoutMs);

                this.#pending.set(requestId, { resolve, reject, timer });

                worker.postMessage({
                    type: requestType,
                    requestId,
                    data: { graphs: chunk, startIndex: start },
                });
            });
        });

        const chunks = await Promise.all(promises);
        return chunks.flat();
    }
}
```

#### Usage Example (Batch Processing)

```javascript
import { MeganWorkerPool } from './megan-pool.js';

// Create pool with 4 workers (or auto-detect CPU count)
const pool = new MeganWorkerPool(4);
await pool.init('https://cdn.example.com/megan/');

// Prepare a batch of graphs
const graphs = molecules.map(mol => ({
    node_attr: mol.nodeFeatures,        // Float32Array
    edge_attr: mol.edgeFeatures,        // Float32Array
    edge_indices: mol.edgeIndices,      // Uint32Array
    num_nodes: mol.numNodes,
    num_edges: mol.numEdges,
}));

// Run parallel inference
const results = await pool.forwardBatch(graphs);

// Each result has: prediction, node_importance, node_importance_norm,
// edge_importance, edge_importance_norm, graph_embedding
for (const result of results) {
    console.log('Prediction:', result.prediction);
}

// Clean up when done
pool.terminate();
```

### 10. Explanation Output for Client-Side Heatmap Rendering

The full forward pass returns node and edge importance values that map directly onto the graph structure. These values are suitable for rendering explanation heatmaps in the browser.

#### Data Layout

For a model with `K` explanation channels (typically K=2 for regression: one channel for positive contributions, one for negative):

| Output | Shape | Layout | Use for heatmap |
|---|---|---|---|
| `node_importance_norm` | (V × K) | `[n * K + k]` = node n, channel k | Color intensity of each node |
| `edge_importance_norm` | (E × K) | `[e * K + k]` = edge e, channel k | Color intensity / width of each edge |
| `graph_embedding` | (K × D) | `[k * D + d]` = channel k, dim d | Not visual; used for similarity search |

The `_norm` variants are normalized to [0, 1] per graph and have a threshold applied (values < 0.1 are dampened to near-zero). These are the values to use for rendering.

#### Extracting Per-Channel Importance

```javascript
/**
 * Extract per-channel importance values from a ForwardResult.
 * Returns an object with separate arrays per channel, ready for rendering.
 *
 * @param {ForwardResult} result
 * @returns {{ nodes: Float32Array[], edges: Float32Array[] }}
 */
function extractChannelImportance(result) {
    const { num_nodes, num_edges, num_channels } = result;
    const nodes = [];
    const edges = [];

    for (let k = 0; k < num_channels; k++) {
        // Node importance for channel k
        const nodeImp = new Float32Array(num_nodes);
        for (let n = 0; n < num_nodes; n++) {
            nodeImp[n] = result.node_importance_norm[n * num_channels + k];
        }
        nodes.push(nodeImp);

        // Edge importance for channel k
        const edgeImp = new Float32Array(num_edges);
        for (let e = 0; e < num_edges; e++) {
            edgeImp[e] = result.edge_importance_norm[e * num_channels + k];
        }
        edges.push(edgeImp);
    }

    return { nodes, edges };
}

// Usage: color nodes by channel 0 (e.g., positive contribution)
const importance = extractChannelImportance(result);
for (let n = 0; n < graph.num_nodes; n++) {
    const intensity = importance.nodes[0][n];  // 0.0 = unimportant, 1.0 = most important
    // Map to color: e.g., rgba(255, 0, 0, intensity)
}
```

### 11. CDN Deployment

The `pkg/` directory output by `compile_gnn_wasm()` contains everything needed for deployment as static files on a CDN.

#### File Structure

```
pkg/
├── megan_wasm_bg.wasm    # Compiled WebAssembly binary (~50-150 KB gzipped)
├── megan_wasm.js         # wasm-pack generated ES module glue (~5 KB)
├── weights.bin           # Model weights (~1-5 MB, depends on architecture)
├── megan-loader.js       # Convenience loader (generated)
├── megan-worker.js       # Web Worker script (generated)
└── megan-pool.js         # Worker pool coordinator (generated)
```

#### Deployment Checklist

**MIME Types** — The server/CDN must serve these content types:

| File | Content-Type |
|---|---|
| `*.wasm` | `application/wasm` |
| `*.js` | `application/javascript` |
| `*.bin` | `application/octet-stream` |

Wasm streaming compilation (`WebAssembly.instantiateStreaming()`) requires `application/wasm`. If the MIME type is wrong, the browser falls back to slower non-streaming compilation.

**Compression** — Enable gzip or brotli compression. Wasm binaries compress extremely well (often 60-70% reduction):

| File | Raw size | Gzipped | Brotli |
|---|---|---|---|
| `megan_wasm_bg.wasm` | ~200 KB | ~60 KB | ~45 KB |
| `weights.bin` (3-layer MEGAN) | ~3 MB | ~2.5 MB | ~2.2 MB |

Weight files compress less well because they're dense floating-point data.

**Caching** — Use aggressive caching for immutable artifacts. Recommended headers:

```
# Wasm binary and weights: immutable, cache for 1 year.
# Use content-hash filenames for cache busting on model updates.
Cache-Control: public, max-age=31536000, immutable

# JS glue: shorter cache (may change with wasm-pack version)
Cache-Control: public, max-age=86400
```

**CORS** — If the wasm module is served from a different origin than the web app, the CDN must include:

```
Access-Control-Allow-Origin: *
```

(Or a specific origin for stricter security.)

**Cross-Origin Isolation** — Web Workers with `SharedArrayBuffer` require COOP/COEP headers. The basic Web Worker approach in this document does NOT require these headers — each worker uses its own ArrayBuffer copy. However, if you later want to share weight memory across workers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

#### Example: Nginx Configuration

```nginx
server {
    listen 443 ssl;
    server_name cdn.example.com;

    # Shared MIME types
    types {
        application/wasm wasm;
        application/javascript js;
        application/octet-stream bin;
    }

    # Immutable assets: wasm binary + weights (content-hashed filenames)
    location ~* ^/megan/.*\.(wasm|bin)$ {
        root /var/www/static;

        gzip on;
        gzip_types application/wasm application/octet-stream;
        gzip_min_length 1024;

        # Brotli (requires ngx_brotli module — better compression for wasm)
        # brotli on;
        # brotli_types application/wasm application/octet-stream;

        add_header Cache-Control "public, max-age=31536000, immutable";
        add_header Access-Control-Allow-Origin *;
    }

    # JS files: shorter cache (may change with code updates)
    location ~* ^/megan/.*\.js$ {
        root /var/www/static;

        gzip on;
        gzip_types application/javascript;
        gzip_min_length 1024;

        add_header Cache-Control "public, max-age=86400";
        add_header Access-Control-Allow-Origin *;
    }

    # Manifest and other metadata
    location ~* ^/megan/.*\.json$ {
        root /var/www/static;
        add_header Cache-Control "public, max-age=300";  # 5 min
        add_header Access-Control-Allow-Origin *;
    }
}
```

#### Example: Cloudflare Pages / S3 + CloudFront

For S3 + CloudFront, set the Content-Type metadata on each object at upload time:

```bash
# Upload script
aws s3 cp pkg/megan_wasm_bg.wasm s3://bucket/megan/ \
    --content-type "application/wasm" \
    --cache-control "public, max-age=31536000, immutable"

aws s3 cp pkg/weights.bin s3://bucket/megan/ \
    --content-type "application/octet-stream" \
    --cache-control "public, max-age=31536000, immutable"

aws s3 cp pkg/megan_wasm.js s3://bucket/megan/ \
    --content-type "application/javascript" \
    --cache-control "public, max-age=86400"

aws s3 cp pkg/megan-loader.js s3://bucket/megan/ \
    --content-type "application/javascript" \
    --cache-control "public, max-age=86400"

aws s3 cp pkg/megan-worker.js s3://bucket/megan/ \
    --content-type "application/javascript" \
    --cache-control "public, max-age=86400"

aws s3 cp pkg/megan-pool.js s3://bucket/megan/ \
    --content-type "application/javascript" \
    --cache-control "public, max-age=86400"
```

### 12. Content Security Policy (CSP)

Loading and compiling WebAssembly requires specific CSP directives. If the consuming web application has a strict Content Security Policy (common in pharmaceutical/enterprise settings), the wasm module will **silently fail** to instantiate without the correct directives.

**Minimum required CSP:**

```
Content-Security-Policy:
    script-src 'self' 'wasm-unsafe-eval' https://cdn.example.com;
    worker-src 'self' blob:;
    connect-src 'self' https://cdn.example.com;
```

Key points:
- **`'wasm-unsafe-eval'`** is required for `WebAssembly.compile()` / `WebAssembly.instantiate()`. This is NOT the same as `'unsafe-eval'` — it only allows wasm compilation, not JavaScript `eval()`.
- **`worker-src`** is needed for the Web Worker pool.
- **`connect-src`** is needed for `fetch()` of the wasm binary and weights.
- `wasm-bindgen` does NOT use `eval()`, so `'unsafe-eval'` is NOT required.

### 13. Monitoring and Observability

In production, wasm module failures are invisible unless explicitly instrumented. The loader should report key events to an error tracking service (e.g., Sentry).

**Key metrics to capture via `performance.mark()` / `performance.measure()`:**

| Metric | What it measures |
|---|---|
| `megan.wasm_compile` | Time to compile/instantiate the wasm module |
| `megan.weights_download` | Time to fetch weights.bin from CDN |
| `megan.weights_load` | Time to deserialize weights into wasm memory |
| `megan.first_inference` | Latency of the first forward() call (includes JIT warmup) |

**Error events to report:**

| Error | Cause |
|---|---|
| Wasm compile failure | CSP blocking, corrupted binary, unsupported browser |
| Weight CRC mismatch | Truncated download, CDN serving stale cache |
| Weight fetch failure | Network error, CORS misconfiguration |
| Worker crash | OOM, Rust panic, browser tab pressure |
| Graph size exceeded | Input exceeds MAX_SUPPORTED_NODES/EDGES |

The `console_error_panic_hook` crate catches Rust panics in the browser console, but this is invisible in production. Wrap the `MeganModel.load()` and `forward()` calls with structured error reporting.

### 14. Verification

Before trusting the compiled model, verify numerical equivalence against PyTorch. The verification runs in Python (or Node.js) against the original model.

#### Python-Side Verification (Against PyTorch)

```python
import numpy as np
import subprocess
import json


def verify_wasm_model(
    original_model,
    pkg_dir: str,
    test_graphs: list[dict],
    rtol: float = 1e-4,
    atol: float = 1e-5,
):
    """
    Run both models on the same inputs and assert all outputs match
    within floating-point tolerance.
    
    Uses Node.js to execute the wasm model (requires Node 16+).
    """
    original_model.eval()
    
    for i, g in enumerate(test_graphs):
        # PyTorch forward pass
        torch_info = original_model.forward_graph(g)
        
        # Wasm forward pass (via Node.js subprocess)
        wasm_info = _run_wasm_forward(pkg_dir, g)
        
        for key in ['graph_output', 'node_importance', 'node_importance_norm',
                     'edge_importance', 'edge_importance_norm']:
            torch_val = torch_info[key]
            wasm_val = wasm_info[key]
            
            max_diff = np.max(np.abs(torch_val - wasm_val))
            if max_diff > atol:
                print(f"Graph {i}, key '{key}': max diff = {max_diff:.2e}")
            
            np.testing.assert_allclose(
                wasm_val, torch_val, rtol=rtol, atol=atol,
                err_msg=f"Mismatch in graph {i}, key '{key}'"
            )
    
    print(f"Verification passed on {len(test_graphs)} graphs!")


def _run_wasm_forward(pkg_dir: str, graph: dict) -> dict:
    """
    Execute the wasm model via Node.js and return the result as numpy arrays.
    """
    # Serialize graph to JSON for the Node.js subprocess
    graph_json = {
        'node_attr': graph['node_attributes'].astype(np.float32).tolist(),
        'edge_attr': graph['edge_attributes'].astype(np.float32).tolist(),
        'edge_indices': graph['edge_indices'].astype(np.int32).tolist(),
        'num_nodes': int(graph['node_attributes'].shape[0]),
        'num_edges': int(graph['edge_indices'].shape[0]),
    }
    
    node_script = f"""
    const {{ readFileSync }} = require('fs');
    const {{ WASI }} = require('wasi');
    
    async function main() {{
        // Load wasm module
        const wasmPath = '{pkg_dir}/megan_wasm_bg.wasm';
        const wasmBuffer = readFileSync(wasmPath);
        const {{ default: init, WasmMegan }} = await import('{pkg_dir}/megan_wasm.js');
        await init();
        
        // Load weights
        const weightsBuffer = readFileSync('{pkg_dir}/weights.bin');
        const model = new WasmMegan(weightsBuffer);
        
        // Run forward pass
        const graph = {json.dumps(graph_json)};
        const result = model.forward(
            new Float32Array(graph.node_attr.flat()),
            new Float32Array(graph.edge_attr.flat()),
            new Uint32Array(graph.edge_indices.flat()),
            graph.num_nodes,
            graph.num_edges,
        );
        
        console.log(JSON.stringify({{
            graph_output: Array.from(result.prediction),
            node_importance: Array.from(result.node_importance),
            node_importance_norm: Array.from(result.node_importance_norm),
            edge_importance: Array.from(result.edge_importance),
            edge_importance_norm: Array.from(result.edge_importance_norm),
        }}));
    }}
    
    main().catch(e => {{ console.error(e); process.exit(1); }});
    """
    
    result = subprocess.run(
        ['node', '--experimental-wasm-modules', '-e', node_script],
        capture_output=True, text=True,
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Node.js verification failed:\n{result.stderr}")
    
    data = json.loads(result.stdout)
    return {k: np.array(v, dtype=np.float32) for k, v in data.items()}


def verify_edge_cases(original_model, pkg_dir: str):
    """Test edge cases that are easy to get wrong in wasm."""
    
    # 1. Single-node graph (no edges)
    g_single = {
        'node_attributes': np.random.randn(1, 9).astype(np.float32),
        'edge_attributes': np.zeros((0, 4), dtype=np.float32),
        'edge_indices': np.zeros((0, 2), dtype=np.int64),
    }
    
    # 2. Graph with inf in node features
    g_inf = {
        'node_attributes': np.array(
            [[np.inf, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0]],
            dtype=np.float32,
        ),
        'edge_attributes': np.random.randn(2, 4).astype(np.float32),
        'edge_indices': np.array([[0, 1], [1, 0]], dtype=np.int64),
    }
    
    # 3. Two-node graph (minimal edges)
    g_minimal = {
        'node_attributes': np.random.randn(2, 9).astype(np.float32),
        'edge_attributes': np.random.randn(2, 4).astype(np.float32),
        'edge_indices': np.array([[0, 1], [1, 0]], dtype=np.int64),
    }
    
    # 4. Large graph that exceeds initial scratch buffer (128 nodes)
    # This tests the ensure_capacity growth logic.
    n_large = 200
    e_large = 600
    g_large = {
        'node_attributes': np.random.randn(n_large, 9).astype(np.float32),
        'edge_attributes': np.random.randn(e_large, 4).astype(np.float32),
        'edge_indices': np.column_stack([
            np.random.randint(0, n_large, e_large),
            np.random.randint(0, n_large, e_large),
        ]).astype(np.int64),
    }
    
    for name, g in [("single-node", g_single), ("inf-features", g_inf),
                     ("minimal", g_minimal), ("large-graph", g_large)]:
        try:
            torch_pred = original_model.predict_graph(g)
            wasm_pred = np.array(
                _run_wasm_forward(pkg_dir, g)['graph_output']
            )
            max_diff = np.max(np.abs(torch_pred - wasm_pred))
            print(f"  {name}: max diff = {max_diff:.2e}")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
```

Note: small numerical differences (1e-5 to 1e-6) are expected because:
- BN folding changes the order of operations slightly
- `matrixmultiply` may use different accumulation order than PyTorch's BLAS backend
- Wasm floating-point semantics are IEEE 754 compliant but may differ in intermediate rounding from x86 extended precision
- LayerNorm computed in a single pass vs PyTorch's two-pass algorithm

Run verification on 1000+ molecules to catch rare outliers from accumulated error across layers.

## Browser Compatibility

| Feature | Chrome | Firefox | Safari | Edge |
|---|---|---|---|---|
| WebAssembly | 57+ | 52+ | 11+ | 16+ |
| Wasm SIMD | 91+ | 89+ | 16.4+ | 91+ |
| ES Module Workers | 80+ | 114+ | 15+ | 80+ |
| `fetch()` streaming | 43+ | 65+ | 10.1+ | 14+ |

**SIMD fallback**: If you need to support browsers without wasm SIMD (rare as of 2026), compile two versions:

```python
# In compile_gnn_wasm():
# Build with SIMD (primary)
compile_gnn_wasm(model, output_dir, enable_simd=True)

# Build without SIMD (fallback)
compile_gnn_wasm(model, f"{output_dir}_no_simd", enable_simd=False)
```

Then feature-detect at load time:

```javascript
async function loadModel(baseUrl) {
    const simdSupported = await wasmSimdSupported();
    const url = simdSupported ? baseUrl : baseUrl + '_no_simd';
    return MeganModel.load({ baseUrl: url });
}

async function wasmSimdSupported() {
    try {
        // Test SIMD support with a minimal wasm module that uses v128
        const bytes = new Uint8Array([
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, 0x03,
            0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01, 0x08, 0x00,
            0xfd, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x0b,
        ]);
        await WebAssembly.compile(new WebAssembly.Module(bytes));
        return true;
    } catch {
        return false;
    }
}
```

## Implementation Roadmap

### Milestone 1: Core Ops Proof of Concept in Wasm

**Goal**: Validate that the core numeric operations produce correct results when compiled to wasm32, and measure baseline performance in the browser.

**Tasks**:
1. Create a Rust crate with `ndarray`, `matrixmultiply`, and `wasm-bindgen` dependencies
2. Implement core ops: `linear`, `linear_leaky_relu`, `scatter_sum`, `scatter_max`, `gather_concat_with_neighbors`, `sigmoid_inplace`
3. Expose a single attention layer forward pass via `wasm-bindgen`
4. Build with `wasm-pack build --target web`
5. Write a minimal HTML page that loads the wasm, passes test data, and logs the output
6. Compare output against PyTorch for 10 test molecules
7. Benchmark: time the wasm attention layer vs PyTorch in the browser console

**Expected outcome**: Numerical match within 1e-4. Measurable speedup even for a single layer. Validates that `matrixmultiply` works correctly on wasm32.

### Milestone 2: Full Forward Pass (Single Molecule)

**Goal**: Complete MEGAN inference in wasm, with all outputs (predictions + explanations).

**Tasks**:
1. Implement the full forward pass: embedding → encoder layers × channels → importance computation → normalization → weighted pooling → projection → final MLP
2. Implement weight serialization (Python side) and loading from `ArrayBuffer` (Rust/wasm side)
3. Implement BN folding in Python, verify fused weights produce correct output
4. Implement the `WasmMegan` wasm-bindgen struct with `forward()` and `predict()` methods
5. Build the `megan-loader.js` convenience wrapper
6. Verify against PyTorch on 1000+ molecules
7. Test edge cases (zero-edge graphs, single-node graphs, inf-in-features)
8. Benchmark: single molecule forward pass in Chrome, Firefox, Safari

**Expected outcome**: Full numerical equivalence. Sub-millisecond single-molecule inference in modern browsers.

### Milestone 3: Code Generation

**Goal**: Generalize from hard-coded architecture to any MEGAN V2 configuration.

**Tasks**:
1. Write Jinja2 templates for the wasm Rust forward pass, parameterized by: number of layers, units per layer, hidden_dim, num_channels, attention_aggregation, projection_units, final_units, prediction_mode, normalize_embedding, output_norm
2. Templates must emit `// [template: section_name]` comments for traceability
3. Implement `extract_architecture()` and `compile_gnn_wasm()` end-to-end
4. Template output snapshot tests: render for 2-3 canonical configs, snapshot the generated Rust source
5. Test with at least 3 different model configurations
6. Add classification model support

**Expected outcome**: `compile_gnn_wasm(model, output_dir)` produces working wasm + JS for any MEGAN V2 configuration.

### Milestone 4: Web Worker Pool for Batch Processing

**Goal**: Parallel batch inference in the browser using Web Workers.

**Tasks**:
1. Implement `megan-worker.js` (worker script that loads its own wasm instance)
2. Implement `MeganWorkerPool` coordinator with `init()`, `forwardBatch()`, `predictBatch()`, `terminate()`
3. Handle worker lifecycle: creation, initialization, error handling, cleanup
4. Test with batch sizes of 32, 100, 1000 molecules
5. Benchmark scaling: 1 worker vs 2, 4, 8 workers
6. Test on Chrome, Firefox, Safari

**Expected outcome**: Near-linear speedup with worker count up to hardware concurrency. Clean API for batch use cases.

### Milestone 5: CDN Deployment Pipeline

**Goal**: One-command deployment of compiled model to a CDN.

**Tasks**:
1. Generate all deployment artifacts in `pkg/`: wasm, JS, weights, loader, worker, pool
2. Add MIME type and caching metadata to generated files
3. Write upload scripts for common CDNs (S3 + CloudFront, Cloudflare Pages, Nginx)
4. Add content-hash filenames for cache busting
5. Test the full flow: compile → upload → load in browser from CDN
6. Add SIMD feature detection and fallback support

**Expected outcome**: `compile_gnn_wasm(model, output_dir)` followed by `./deploy.sh` produces a production-ready CDN deployment.

## Key Technical Decisions

### f32 throughout
Use **f32** for all computation. PyTorch models are typically trained in f32, and wasm's native float type is f32. This halves memory bandwidth and maximizes SIMD throughput (4 floats per 128-bit v128 lane vs 2 doubles).

### matrixmultiply instead of BLAS
Use the **`matrixmultiply`** crate. This is not optional for wasm — BLAS libraries (OpenBLAS, MKL) cannot be compiled to wasm32 because they depend on OS threading, system allocators, and platform-specific assembly. `matrixmultiply` is pure Rust and compiles to wasm without modification. For molecular-sized matrices, it's also faster than BLAS due to lower dispatch overhead.

### wasm SIMD
Compile with `RUSTFLAGS="-C target-feature=+simd128"` to enable 128-bit wasm SIMD. The Rust compiler auto-vectorizes simple scalar loops (sigmoid, activation, element-wise ops) using wasm v128 instructions. Note that `matrixmultiply`'s hand-tuned SIMD kernels only target x86 (SSE/AVX/FMA) and aarch64 (NEON) — on wasm32, it falls back to a generic scalar kernel that LLVM may partially auto-vectorize. This is 128-bit (equivalent to SSE2), narrower than AVX2 (256-bit) on native x86. For molecular GNNs, the impact is modest because the bottleneck is scatter/gather memory access patterns, not vectorized compute.

### Web Workers for parallelism (not SharedArrayBuffer threads)
Wasm threads via `SharedArrayBuffer` require COOP/COEP headers on the serving domain, which can break third-party embeds (iframes, analytics scripts). Web Workers are universally supported and don't require special headers. The trade-off: each worker has its own copy of weights in its own wasm linear memory. For MEGAN models (~1-5 MB weights), having 4-8 copies is acceptable. If memory becomes a concern, `SharedArrayBuffer` for read-only weight sharing is a future optimization.

### No rayon
Rayon requires OS-level threads. Wasm has no native threading (the `wasm32-unknown-unknown` target has no `std::thread`). Parallelism is achieved at the JavaScript level via Web Workers, each running its own wasm instance. This actually simplifies the Rust code — no `Sync`/`Send` bounds, no thread-local scratch management.

### Weight loading via HTTP fetch
Weights are loaded as a separate `weights.bin` file fetched over HTTP, not embedded in the wasm binary via `include_bytes!()`. Reasons:
- **Caching**: The browser caches `weights.bin` independently. Updating JS code doesn't re-download weights.
- **Compile time**: `include_bytes!()` on a 5 MB file dramatically slows Rust compilation.
- **Binary size**: A wasm binary with embedded weights is too large for streaming compilation.
- **Progress tracking**: HTTP fetch supports `ReadableStream` for download progress UI.

### ES Module target
Use `wasm-pack build --target web` which produces ES modules. This works with `<script type="module">`, modern bundlers (Vite, Rollup, webpack 5), and module Web Workers. The alternative `--target bundler` is webpack-specific and less portable.

### Binary size optimization
The release profile uses `lto = true`, `codegen-units = 1`, `strip = "symbols"`, and `panic = "abort"` (saves 10-30 KB by removing unwinding machinery) to minimize wasm binary size. `wasm-pack` automatically runs `wasm-opt` (with `-O3`) on the output — this typically reduces binary size by another 10-20% and can improve runtime performance. Realistic binary size for a MEGAN model with `ndarray` dependency: **200-400 KB raw, 100-200 KB gzipped**. The `ndarray` crate adds significant generic machinery; if binary size is critical, it can be replaced with flat `Vec<f32>` and manual stride arithmetic in a future optimization pass.

## Explicit Out-of-Scope

To prevent scope creep, the following are **not** part of this project:

- **Server-side rendering of heatmaps**: Heatmap rendering is done entirely client-side using the importance outputs from the wasm module. No server-side rendering pipeline is needed.
- **SMILES parsing in wasm**: The API accepts pre-computed graph dicts. SMILES-to-graph conversion can be done client-side via `rdkit-js` or server-side before sending to the browser, but is not part of this module.
- **Layer variants V1 and V3**: Start with V2 only (the default).
- **Training / backpropagation**: This is inference only.
- **GPU via WebGPU**: WebGPU compute shaders could accelerate large-batch inference but are a separate project with different complexity.
- **Quantization / float16 / int8 inference**: Tempting optimization, massive complexity. Note: the CDN deployment section previously mentioned float16 weight quantization as having "negligible accuracy impact" — this claim is unsubstantiated. BN-folded weights can have large dynamic range, and float16 has only ~3 decimal digits of precision. Any quantization approach requires empirical validation against the full verification suite before deployment.
- **MVE (Mean Variance Estimation)**: Rarely used optional feature.
- **SharedArrayBuffer weight sharing**: The basic Web Worker approach copies weights per worker. Shared memory is a future optimization.
- **Service Worker offline caching**: The browser's HTTP cache handles this adequately for CDN-hosted static files.
- **Framework-specific wrappers** (React hooks, Vue composables): The API is vanilla JS. Framework wrappers can be built on top.
