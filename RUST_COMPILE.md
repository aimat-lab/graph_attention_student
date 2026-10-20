# Compiling GNN Models to Rust for Fast CPU Inference

Design document for a system that takes a trained PyTorch GNN model and compiles it into a native Rust shared library for high-performance CPU inference.


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

### Why this works

PyTorch Geometric on CPU spends most of its time on overhead, not on actual math — especially for small graphs like molecules (10–50 nodes, 20–100 edges). The per-operation overhead includes:

- Python interpreter dispatch between every operation
- PyTorch's C++ dispatcher + operator lookup (even with `torch.no_grad()`)
- A new tensor allocation for every intermediate result (gather, concat, linear, batchnorm, activation, scatter — each one is a separate allocation + kernel)
- PyG's Python-level `MessagePassing` plumbing: `propagate()` → `message()` → `aggregate()` → `update()`, each a Python method call
- `scatter_add` / `scatter_max` on CPU: generic implementations that handle arbitrary dtypes, dims, and shapes — massive overhead for the tiny per-node aggregations in molecular graphs

For a typical MEGAN layer with a 30-atom molecule: the actual compute is ~74K MACs per edge × 30 edges ≈ 2.2M MACs. A modern CPU can do this in microseconds. But PyG takes hundreds of microseconds to milliseconds in dispatch and allocation alone.

### Expected speedup

| Scenario | Bottleneck | Expected speedup over PyG CPU |
|---|---|---|
| Single molecule inference | Framework overhead dominates | **30–100x** |
| Small batch (32 molecules) | Mixed overhead/compute | **10–30x** |
| Large batch (1000 molecules) | Matmul compute starts to matter | **2–8x** |

The large-batch number is lower because at batch=1000, PyG batches all edges into one matrix (e.g., `(30000, 132) @ (132, 128)`) and MKL handles that GEMM efficiently. But even there, 2–8x comes from:

- **Op fusion**: PyG runs ~15 separate kernels per layer (gather, concat, linear, batchnorm, activation, ...). Each is a separate tensor allocation + dispatch. Rust fuses these into one loop.
- **Scatter on CPU**: PyG's `scatter_add`/`scatter_max` is generic and slow for small irregular groups (few edges per node). Iterating per-node over a CSR neighbor list is dramatically better for cache locality.
- **BatchNorm folding**: In inference mode, BN is `(x - mean) * scale + bias` — a trivial fused multiply-add that doesn't need its own kernel.

### Where the gains are smaller

- **Very large graphs** (thousands of nodes) with wide feature dimensions: the matmul itself becomes the bottleneck and optimized BLAS is already near-optimal.
- **Attention-heavy architectures** where matmuls dominate over scatter/gather.
- This is **inference only**. Implementing backprop in Rust is not worth the effort.

## Architecture Overview

```
Python compile() call
    │
    ├── 0. Pre-flight check (cargo, rustc version, system deps)
    ├── 1. Introspect PyTorch model → extract architecture graph + weight tensors
    ├── 2. Apply compile-time optimizations (BN folding into linear layers)
    ├── 3. Serialize optimized weights to binary file (with magic + version + checksum)
    ├── 4. Generate Rust source code from Jinja2 templates
    ├── 5. RUSTFLAGS="-C target-cpu=native" cargo build --release → .so
    └── 6. Return Python handle that loads the .so via PyO3
```

The system has two Rust components:

1. **`gnn-runtime`** — a reusable crate with optimized building blocks (fused linear+BN+activation, scatter ops, gather+concat, etc.). Uses the `matrixmultiply` crate for matmuls instead of full BLAS — lower dispatch overhead for the small matrices typical in molecular GNNs.
2. **Generated model crate** — auto-generated Rust code that wires the runtime ops together for a specific model architecture, depends on `gnn-runtime`

## Component Details

### 1. Python Compiler (`compile()` function)

The entry point. Validates the environment, walks the PyTorch module tree, extracts architecture info and weights, generates Rust code, and compiles it.

```python
from jinja2 import Template
import subprocess
import shutil
import struct
import numpy as np
import torch
import torch.nn as nn


class CompileError(Exception):
    """Raised when the Rust compilation pipeline fails."""
    pass


def _check_rust_toolchain(min_version: str = "1.70.0"):
    """
    Verify cargo and rustc are installed and meet minimum version.
    Raises CompileError with install instructions if not found.
    """
    for cmd in ["cargo", "rustc"]:
        if shutil.which(cmd) is None:
            raise CompileError(
                f"'{cmd}' not found. Install the Rust toolchain: "
                "https://rustup.rs/ — then run: rustup default stable"
            )
    
    result = subprocess.run(
        ["rustc", "--version"], capture_output=True, text=True
    )
    # Parse version and compare against min_version
    version_str = result.stdout.strip().split()[1]
    if version_str < min_version:
        raise CompileError(
            f"rustc {version_str} found, but >= {min_version} required. "
            "Run: rustup update stable"
        )


def compile_gnn(
    model: nn.Module,
    output_dir: str,
    model_name: str = "compiled_model",
) -> "CompiledModel":
    """
    Compile a trained PyTorch GNN model into a native Rust shared library.
    
    Args:
        model: Trained PyTorch model in eval mode.
        output_dir: Directory where the Rust project and compiled artifacts are written.
        model_name: Name for the generated crate and Python-loadable module.
        
    Returns:
        A CompiledModel handle that can be used for inference.
        
    Raises:
        CompileError: If Rust toolchain is missing or compilation fails.
    """
    # Step 0: Pre-flight check
    _check_rust_toolchain()
    
    model.eval()
    
    # Step 1: Extract architecture as a list of operation descriptors
    arch = extract_architecture(model)
    
    # Step 2: Fold BatchNorm into preceding linear layers (compile-time optimization)
    state = fold_batchnorms(model)
    
    # Step 3: Also extract LayerNorm parameters (cannot be folded — runtime op)
    layer_norm_params = extract_layer_norms(model)
    state.update(layer_norm_params)
    
    # Step 4: Serialize weights with magic + version + checksum
    write_weights_bin(state, f"{output_dir}/weights.bin")
    
    # Step 5: Generate Rust source from templates
    rust_code = FORWARD_TEMPLATE.render(arch=arch, model_name=model_name)
    write_file(f"{output_dir}/src/lib.rs", rust_code)
    write_file(f"{output_dir}/Cargo.toml", CARGO_TEMPLATE.render(name=model_name))
    
    # Step 6: Compile with native CPU target for SIMD auto-vectorization
    import os
    env = os.environ.copy()
    env["RUSTFLAGS"] = "-C target-cpu=native"
    
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=output_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CompileError(
            f"Rust compilation failed:\n{result.stderr}\n"
            f"Generated source at: {output_dir}/src/lib.rs"
        )
    
    # Step 7: Return handle
    return CompiledModel(f"{output_dir}/target/release/lib{model_name}.so",
                         f"{output_dir}/weights.bin")
```

### 2. Architecture Extraction

Walk the module tree and build a representation of the computation graph that the code generator can consume.

```python
def extract_architecture(model: nn.Module) -> list[dict]:
    """
    Extract a layer-by-layer description of the model architecture.
    
    Returns a list of operation descriptors, each a dict like:
        {"op": "linear", "name": "lay_embedd", "in_dim": 9, "out_dim": 64}
        {"op": "attention_layer", "name": "...", "variant": "v2",
         "in_dim": 64, "out_dim": 64, "edge_dim": 4, "hidden_dim": 128}
    
    Also extracts model-level config:
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
    
    # Extract model-level config needed by the code generator
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

This is a major free win. At compile time, fold each BatchNorm's parameters into the preceding linear layer's weight and bias. This eliminates BatchNorm as a runtime operation entirely.

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
    
    # Find (Linear, BatchNorm1d) pairs in Sequential containers
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
                        state[f"{linear_name}.weight"] = linear.weight.detach().cpu().numpy()
                        if linear.bias is not None:
                            state[f"{linear_name}.bias"] = linear.bias.detach().cpu().numpy()
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
    b = linear.bias.data.clone() if linear.bias is not None else torch.zeros(w.shape[0])
    
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

Simple binary format with magic number, version, and CRC32 checksum for corruption detection.

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


def read_weights_bin(path: str) -> dict[str, np.ndarray]:
    """Load weights back with integrity verification (useful for Python-side verification)."""
    with open(path, 'rb') as f:
        data = f.read()
    
    # Verify checksum (last 4 bytes)
    stored_checksum = struct.unpack('<I', data[-4:])[0]
    computed_checksum = zlib.crc32(data[:-4]) & 0xFFFFFFFF
    if stored_checksum != computed_checksum:
        raise ValueError(f"Weight file corrupted: CRC32 mismatch "
                        f"(stored={stored_checksum:#x}, computed={computed_checksum:#x})")
    
    offset = 0
    magic = data[offset:offset+4]; offset += 4
    if magic != WEIGHTS_MAGIC:
        raise ValueError(f"Not a GNN weights file (bad magic: {magic!r})")
    
    version = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
    if version != WEIGHTS_VERSION:
        raise ValueError(f"Unsupported weight format version: {version}")
    
    count = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
    
    arrays = {}
    for _ in range(count):
        name_len = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
        name = data[offset:offset+name_len].decode('utf-8'); offset += name_len
        
        ndim = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack('<I', data[offset:offset+4])[0]); offset += 4
        shape = tuple(shape)
        
        numel = 1
        for s in shape:
            numel *= s
        arr = np.frombuffer(data[offset:offset+numel*4], dtype=np.float32).reshape(shape)
        offset += numel * 4
        arrays[name] = arr.copy()  # copy because frombuffer returns read-only view
    
    return arrays
```

### 5. Rust Runtime Crate (`gnn-runtime`)

This crate provides the optimized building blocks. Uses `matrixmultiply` instead of full BLAS (via ndarray-linalg) because for the small matrix sizes in molecular GNNs (~30 rows), full BLAS has significant dispatch overhead. The `matrixmultiply` crate provides a lightweight, zero-overhead matmul that auto-vectorizes with AVX2/AVX-512 when compiled with `-C target-cpu=native`.

**Cargo.toml dependencies:**
```toml
[dependencies]
ndarray = "0.16"
matrixmultiply = "0.3"
rayon = "1"
pyo3 = { version = "0.22", features = ["extension-module"] }
numpy = "0.22"
```

Note: no `ndarray-linalg` dependency. This eliminates the fragile BLAS backend linking (OpenBLAS/MKL feature flags, libgfortran dependencies, thread pool conflicts with rayon). The `matrixmultiply` crate is pure Rust with hand-tuned SIMD kernels and zero system dependencies.

#### Core Operations

```rust
// gnn_runtime/src/ops.rs

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
    
    // output = input @ weight^T
    // matrixmultiply expects column-major strides, but we work in row-major.
    // Trick: C = A @ B^T  is equivalent to  C^T = B @ A^T
    // We compute in-place with the row-major layout.
    unsafe {
        matrixmultiply::sgemm(
            n, m, k,           // rows of C, cols of C, inner dim
            1.0,               // alpha
            input.as_ptr(), input.strides()[0] as isize, input.strides()[1] as isize,
            weight.as_ptr(), weight.strides()[1] as isize, weight.strides()[0] as isize, // transposed
            0.0,               // beta
            output.as_mut_ptr(), output.strides()[0] as isize, output.strides()[1] as isize,
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
    input: &ArrayView2<f32>,      // (N, in_dim)
    weight: &ArrayView2<f32>,     // (out_dim, in_dim)
    bias: &Array1<f32>,           // (out_dim,)
    negative_slope: f32,
    output: &mut ArrayViewMut2<f32>, // (N, out_dim) — pre-allocated
) {
    matmul_add_bias(input, weight, bias, output);
    
    // Fused activation (no intermediate tensor)
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
        let var: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d;
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
/// This implements SumNeighbors (layers.py:307-329).
///
/// The neighbor sum for node i = sum of node_features[src[e]] for all edges where dst[e] == i.
/// This is because PyG's MessagePassing sends x_j (source) and aggregates at destination.
pub fn neighbor_sum(
    node_features: &ArrayView2<f32>,  // (V, F)
    src: &[u32],                      // (E,) source node per edge
    dst: &[u32],                      // (E,) destination node per edge
    num_nodes: usize,
    output: &mut ArrayViewMut2<f32>,  // (V, F) — pre-allocated
) {
    let nf = output.slice_mut(s![..num_nodes, ..]);
    nf.fill(0.0);
    for e in 0..src.len() {
        let s = src[e] as usize;
        let d = dst[e] as usize;
        // Gather source features and scatter-add to destination
        for f in 0..node_features.ncols() {
            output[[d, f]] += node_features[[s, f]];
        }
    }
}

/// Fused gather source + destination node features and concatenate with edge features.
/// This replaces three separate PyG operations (index_select × 2 + cat) with one loop.
/// 
/// For V1/V3 layers: output row = [x_src, x_dst, edge_attr]
pub fn gather_concat(
    node_features: &ArrayView2<f32>,   // (V, F_node)
    edge_features: &ArrayView2<f32>,   // (E, F_edge)
    src: &[u32],
    dst: &[u32],
    output: &mut ArrayViewMut2<f32>,   // (E, 2*F_node + F_edge) — pre-allocated
) {
    let f = node_features.ncols();
    for e in 0..src.len() {
        let s = src[e] as usize;
        let d = dst[e] as usize;
        let mut row = output.row_mut(e);
        row.slice_mut(s![..f]).assign(&node_features.row(s));
        row.slice_mut(s![f..2*f]).assign(&node_features.row(d));
        row.slice_mut(s![2*f..]).assign(&edge_features.row(e));
    }
}

/// Variant for V2 layers that includes pre-computed neighbor-sum features.
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
/// Uses sliced view — only zeros and writes to output[..num_nodes].
pub fn scatter_sum(
    edge_values: &ArrayView2<f32>,   // (E, D)
    dst: &[u32],                     // (E,) destination node for each edge
    num_nodes: usize,
    output: &mut ArrayViewMut2<f32>, // (V, D) — pre-allocated, only [..num_nodes] used
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
/// Used for edge→node importance pooling.
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

#### Pre-Allocated Scratch Buffers

Allocate all intermediate buffers once at initialization, reuse them across calls. This is the single biggest source of speedup for small graphs — PyG allocates fresh tensors for every operation.

All ops take sliced views (`s![..num_edges, ..]`) of oversized buffers so that `fill(0.0)` only touches rows actually needed.

```rust
// gnn_runtime/src/scratch.rs

use ndarray::Array2;

/// Pre-allocated buffers for a single-molecule forward pass.
/// For per-molecule parallelism, each rayon thread gets its own Scratch
/// indexed by rayon::current_thread_index().
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
    
    // Per-channel node embeddings (avoid repeated allocation in forward loop)
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
    pub graph_embedding: Array2<f32>,  // (1, embed_dim) — single molecule
    pub projection_buf: Array2<f32>,   // (1, proj_dim) — for projection MLP
    
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
    pub fn ensure_capacity(&mut self, num_nodes: usize, num_edges: usize, dims: &ModelDims) {
        if num_edges > self.edge_concat.nrows() {
            let new_cap = (num_edges * 3) / 2;  // grow by 1.5x to avoid frequent realloc
            self.edge_concat = Array2::zeros((new_cap, dims.concat_dim));
            self.msg_hidden = Array2::zeros((new_cap, dims.hidden_dim));
            self.msg_out = Array2::zeros((new_cap, dims.out_dim));
            self.attn_hidden = Array2::zeros((new_cap, dims.hidden_dim));
            self.attn_scores = Array2::zeros((new_cap, 1));
            self.attn_sigmoid = Array2::zeros((new_cap, 1));
            self.weighted_msg = Array2::zeros((new_cap, dims.out_dim));
            self.edge_importance = Array2::zeros((new_cap, dims.num_channels));
            self.edge_importance_norm = Array2::zeros((new_cap, dims.num_channels));
            // Also grow alpha_storage
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
            self.transform_input = Array2::zeros((new_cap, dims.out_dim + dims.max_in_dim));
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
    }
}
```

### 6. Generated Model Code

The Jinja2 template produces a complete Rust module for a specific model. Here's what the generated code looks like for a MEGAN model (GraphAttentionLayerV2, 3 layers, 64 units, hidden_dim=128, 2 channels).

The generated source includes a preamble comment with the model config and `// [template: ...]` markers at key sections so that Rust compiler errors can be traced back to the template that produced them.

```rust
// Auto-generated by gnn_compiler — do not edit manually
// Model: MEGAN (v2, layers=3, units=[64,64,64], hidden=128, channels=2)
// Attention aggregation: max
// Prediction mode: regression
// Source checkpoint: /path/to/model.ckpt

use gnn_runtime::ops::*;
use gnn_runtime::scratch::Scratch;
use ndarray::prelude::*;
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

// [template: constants]
const NUM_CHANNELS: usize = 2;
const NUM_LAYERS: usize = 3;
const EMBED_DIM: usize = 64;

/// All model weights, loaded from weights.bin at initialization.
struct Weights {
    // Embedding: node_dim -> units[0] (plain linear, NO activation)
    embed_w: Array2<f32>,  // (64, 9)
    embed_b: Array1<f32>,  // (64,)
    
    // Per channel, per layer: attention layer weights
    channels: Vec<ChannelWeights>,
    
    // Projection layers per channel (if any)
    // Each entry: (linear_w, linear_b, layer_norm_w, layer_norm_b)
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
    msg_w1: Array2<f32>,   // NOTE: no activation after this layer in message MLP
    msg_b1: Array1<f32>,
    
    // Attention MLP: (3*in_dim + edge_dim) -> hidden -> hidden -> 1
    attn_w0: Array2<f32>,
    attn_b0: Array1<f32>,
    attn_w1: Array2<f32>,
    attn_b1: Array1<f32>,
    attn_w2: Array2<f32>,  // (1, hidden) — no activation after this
    attn_b2: Array1<f32>,
    
    // Transform MLP: (out + in) -> hidden -> out  (both with BN+LeakyReLU folded)
    trans_w0: Array2<f32>,
    trans_b0: Array1<f32>,
    trans_w1: Array2<f32>,
    trans_b1: Array1<f32>,
}

struct ProjectionLayerParams {
    weight: Array2<f32>,
    bias: Array1<f32>,
    ln_weight: Array1<f32>,   // LayerNorm scale
    ln_bias: Array1<f32>,     // LayerNorm shift
    ln_eps: f32,
}

struct DenseLayerParams {
    weight: Array2<f32>,
    bias: Array1<f32>,
    // For intermediate layers: also has LayerNorm params
    ln_weight: Option<Array1<f32>>,
    ln_bias: Option<Array1<f32>>,
    is_last: bool,
}


/// Forward pass for a SINGLE molecule.
/// Called in parallel via rayon for batch processing.
// [template: forward_single]
fn forward_single(
    weights: &Weights,
    scratch: &mut Scratch,
    node_attr: &ArrayView2<f32>,  // (V, 9)
    edge_attr: &ArrayView2<f32>,  // (E, 4)
    src: &[u32],                  // (E,)
    dst: &[u32],                  // (E,)
) -> SingleForwardResult {
    let num_nodes = node_attr.nrows();
    let num_edges = src.len();
    
    scratch.ensure_capacity(num_nodes, num_edges, &DIMS);
    
    // --- Workaround: replace inf with 0 (matches megan.py:622) ---
    let mut node_input = node_attr.to_owned();
    replace_inf_with_zero(&mut node_input.view_mut());
    
    // --- Embedding (plain linear, NO activation) ---
    // [template: embedding]
    {
        let mut emb_view = scratch.node_emb.slice_mut(s![..num_nodes, ..]);
        linear(
            &node_input.view(), &weights.embed_w.view(), &weights.embed_b,
            &mut emb_view,
        );
    }
    
    // Initialize per-channel embeddings (copy from shared embedding)
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
            
            // Borrow the current channel embedding as a read-only view
            // (we need to read from it while also writing scratch buffers)
            let x = scratch.channel_embs[ch].slice(s![..num_nodes, ..]).to_owned();
            
            // 1. Compute neighbor sum: for each node, sum features of neighbors
            //    neighbor_sum[i] = sum of x[src[e]] for all edges where dst[e] == i
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
            //    lay_message = Linear+BN+LeakyReLU -> Linear (no activation on last)
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
            
            // 4. Attention MLP: Linear+BN+LeakyReLU -> Linear+BN+LeakyReLU -> Linear
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
            
            // 5. Sigmoid attention * message (using pre-allocated sigmoid buffer)
            scratch.attn_sigmoid.slice_mut(s![..num_edges, ..])
                .assign(&scratch.attn_scores.slice(s![..num_edges, ..]));
            sigmoid_inplace(&mut scratch.attn_sigmoid.slice_mut(s![..num_edges, ..]));
            
            for e in 0..num_edges {
                let alpha = scratch.attn_sigmoid[[e, 0]];
                for d in 0..scratch.msg_out.ncols() {
                    scratch.weighted_msg[[e, d]] = alpha * scratch.msg_out[[e, d]];
                }
            }
            
            // 6. Scatter sum: aggregate weighted messages to destination nodes
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
                    scratch.transform_input[[n, d]] = scratch.node_aggregated[[n, d]];
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
            
            // Residual connection is INSIDE GraphAttentionLayerV2 (layers.py:428)
            // node_embedding += x — this is the ONLY residual, do NOT add another
            for n in 0..num_nodes {
                for d in 0..out_dim {
                    scratch.channel_embs[ch][[n, d]] = scratch.node_out[[n, d]] + x[[n, d]];
                }
            }
            
            // Store attention logits for importance computation
            scratch.alpha_storage[layer_idx][ch]
                .slice_mut(s![..num_edges, ..])
                .assign(&scratch.attn_scores.slice(s![..num_edges, ..]));
        }
    }
    
    // --- Importance computation ---
    // [template: importance]
    // Stack across layers, apply sigmoid per layer, then aggregate
    for k in 0..NUM_CHANNELS {
        for e in 0..num_edges {
            // Aggregate sigmoid(logit) across layers
            let agg = match weights.attention_aggregation {
                AttentionAggregation::Max => {
                    let mut val = f32::NEG_INFINITY;
                    for l in 0..NUM_LAYERS {
                        let sig = 1.0 / (1.0 + (-scratch.alpha_storage[l][k][[e, 0]]).exp());
                        val = val.max(sig);
                    }
                    val
                }
                AttentionAggregation::Min => {
                    let mut val = f32::INFINITY;
                    for l in 0..NUM_LAYERS {
                        let sig = 1.0 / (1.0 + (-scratch.alpha_storage[l][k][[e, 0]]).exp());
                        val = val.min(sig);
                    }
                    val
                }
                AttentionAggregation::Sum => {
                    let mut val = 0.0f32;
                    for l in 0..NUM_LAYERS {
                        val += 1.0 / (1.0 + (-scratch.alpha_storage[l][k][[e, 0]]).exp());
                    }
                    val
                }
                AttentionAggregation::Mean => {
                    let mut val = 0.0f32;
                    for l in 0..NUM_LAYERS {
                        val += 1.0 / (1.0 + (-scratch.alpha_storage[l][k][[e, 0]]).exp());
                    }
                    val / NUM_LAYERS as f32
                }
            };
            scratch.edge_importance[[e, k]] = agg;
        }
    }
    
    // Node importance: mean of max-pool from src and dst endpoints (megan.py:688-691)
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
                scratch.node_imp_src[[n, k]] + scratch.node_imp_dst[[n, k]]
            );
        }
    }
    
    // Normalize node importance per graph (single molecule → 1 graph)
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
    
    // Normalize edge importance per graph (megan.py:753-761)
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
    let mut graph_embedding_channels: Vec<Array1<f32>> = Vec::with_capacity(NUM_CHANNELS);
    
    for k in 0..NUM_CHANNELS {
        let mut ge = Array1::zeros(embed_dim);
        for n in 0..num_nodes {
            let weight = scratch.node_importance_norm[[n, k]];
            for d in 0..embed_dim {
                ge[d] += weight * scratch.channel_embs[k][[n, d]];
            }
        }
        
        // Apply projection layers: Linear -> SiLU -> LayerNorm (per layer)
        // [template: projection, ch={k}]
        if !weights.proj[k].is_empty() {
            let mut proj_input = ge.insert_axis(ndarray::Axis(0)); // (1, D)
            
            for proj_params in &weights.proj[k] {
                let mut proj_out = Array2::zeros((1, proj_params.weight.nrows()));
                linear_silu(
                    &proj_input.view(), &proj_params.weight.view(), &proj_params.bias,
                    &mut proj_out.view_mut(),
                );
                layer_norm(
                    &mut proj_out.view_mut(),
                    &proj_params.ln_weight, &proj_params.ln_bias, proj_params.ln_eps,
                );
                proj_input = proj_out;
            }
            
            ge = proj_input.row(0).to_owned();
        }
        
        // Optional L2 normalization (when normalize_embedding=True)
        if weights.normalize_embedding {
            let norm: f32 = ge.iter().map(|&v| v * v).sum::<f32>().sqrt();
            if norm > 1e-12 {
                ge.mapv_inplace(|v| v / norm);
            }
        }
        
        graph_embedding_channels.push(ge);
    }
    
    // --- Concatenate channel embeddings and run final prediction MLP ---
    // [template: final_mlp]
    let total_dim: usize = graph_embedding_channels.iter().map(|e| e.len()).sum();
    let mut concat = Array2::zeros((1, total_dim));
    let mut offset = 0;
    for ge in &graph_embedding_channels {
        for d in 0..ge.len() {
            concat[[0, offset + d]] = ge[d];
        }
        offset += ge.len();
    }
    
    // Final dense layers: Linear -> SiLU -> LayerNorm (intermediate), Linear (last)
    let mut x = concat;
    for params in &weights.final_layers {
        let mut out = Array2::zeros((1, params.weight.nrows()));
        if params.is_last {
            linear(&x.view(), &params.weight.view(), &params.bias, &mut out.view_mut());
            // Add regression reference for regression models
            out.mapv_inplace(|v| v + weights.regression_reference);
            
            // Optional output norm for classification (megan.py:845-846)
            if let Some(norm_val) = weights.output_norm {
                l2_normalize(&mut out.view_mut());
                out.mapv_inplace(|v| v * norm_val);
            }
        } else {
            linear_silu(&x.view(), &params.weight.view(), &params.bias, &mut out.view_mut());
            if let (Some(w), Some(b)) = (&params.ln_weight, &params.ln_bias) {
                layer_norm(&mut out.view_mut(), w, b, 1e-5);
            }
        }
        x = out;
    }
    
    // --- Collect results ---
    SingleForwardResult {
        prediction: x.row(0).to_owned(),
        node_importance: scratch.node_importance.slice(s![..num_nodes, ..]).to_owned(),
        node_importance_norm: scratch.node_importance_norm.slice(s![..num_nodes, ..]).to_owned(),
        edge_importance: scratch.edge_importance.slice(s![..num_edges, ..]).to_owned(),
        edge_importance_norm: scratch.edge_importance_norm.slice(s![..num_edges, ..]).to_owned(),
        graph_embedding: graph_embedding_channels,
    }
}


// --- PyO3 interface ---
// [template: pyo3_bindings]

#[pyclass]
struct CompiledMegan {
    weights: Weights,
    // One Scratch per rayon thread, indexed by thread index.
    // Pre-allocated at init for the rayon thread pool size.
    thread_scratches: Vec<Scratch>,
}

#[pymethods]
impl CompiledMegan {
    #[new]
    fn new(weights_path: &str) -> PyResult<Self> {
        let weights = Weights::load(weights_path)?;
        let num_threads = rayon::current_num_threads();
        let thread_scratches = (0..num_threads)
            .map(|_| Scratch::new(128, 256, &DIMS))
            .collect();
        Ok(Self { weights, thread_scratches })
    }
    
    /// Full forward pass returning all outputs (predictions + explanations).
    /// Accepts zero-copy numpy array views — no input copying.
    fn forward(
        &mut self,
        node_attr: PyReadonlyArray2<f32>,
        edge_attr: PyReadonlyArray2<f32>,
        src: PyReadonlyArray1<u32>,
        dst: PyReadonlyArray1<u32>,
        batch: PyReadonlyArray1<u32>,
        num_graphs: usize,
    ) -> PyResult<PyObject> {
        // Split batched input into per-molecule data, run in parallel
        let molecules = split_batch(
            &node_attr.as_array(),
            &edge_attr.as_array(),
            src.as_slice()?,
            dst.as_slice()?,
            batch.as_slice()?,
            num_graphs,
        );
        
        let results: Vec<SingleForwardResult> = molecules
            .into_par_iter()
            .map(|mol| {
                let thread_idx = rayon::current_thread_index().unwrap_or(0);
                let scratch = &mut self.thread_scratches[thread_idx];
                forward_single(
                    &self.weights, scratch,
                    &mol.node_attr, &mol.edge_attr,
                    &mol.src, &mol.dst,
                )
            })
            .collect();
        
        // Merge results into batched numpy arrays and return as Python dict
        Python::with_gil(|py| merge_results_to_dict(py, &results))
    }
    
    /// Predict only (no explanations). Slightly faster since we could skip
    /// importance computation, but for now just wraps forward().
    fn predict(
        &mut self,
        node_attr: PyReadonlyArray2<f32>,
        edge_attr: PyReadonlyArray2<f32>,
        src: PyReadonlyArray1<u32>,
        dst: PyReadonlyArray1<u32>,
        batch: PyReadonlyArray1<u32>,
        num_graphs: usize,
    ) -> PyResult<Py<PyArray2<f32>>> {
        // Same as forward() but only return predictions
        // ...
    }
}
```

### 7. Python Wrapper (Loading the Compiled Model)

Provides the same interface as the PyTorch `Megan` model: `predict_graph`, `predict_graphs`, `forward_graph`, `forward_graphs`.

```python
import numpy as np


class CompiledModel:
    """
    Wraps a compiled Rust .so and provides the same interface as the PyTorch model.
    All methods return numpy arrays, matching Megan.forward_graph() / predict_graphs().
    """
    
    def __init__(self, lib_path: str, weights_path: str):
        """Load the compiled shared library and initialize the model."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("compiled_model", lib_path)
        self._module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._module)
        
        self._model = self._module.CompiledMegan(weights_path)
    
    def forward_graphs(
        self,
        graphs: list[dict],
        batch_size: int = 1000,
    ) -> list[dict]:
        """
        Full forward pass on a list of GraphDicts.
        Returns list of dicts, one per graph, with keys:
            graph_output, node_importance, node_importance_norm,
            edge_importance, edge_importance_norm, graph_embedding
        Same interface as Megan.forward_graphs().
        """
        all_results = []
        
        for start in range(0, len(graphs), batch_size):
            batch_graphs = graphs[start:start + batch_size]
            node_attr, edge_attr, src, dst, batch_idx, num_graphs = (
                self._batch_graphs(batch_graphs)
            )
            
            # Returns a dict of batched numpy arrays
            batched_result = self._model.forward(
                node_attr, edge_attr, src, dst, batch_idx, num_graphs,
            )
            
            # Unbatch into per-graph dicts
            all_results.extend(
                self._unbatch_results(batched_result, batch_graphs)
            )
        
        return all_results
    
    def forward_graph(self, graph: dict) -> dict:
        """Single graph forward pass."""
        return self.forward_graphs([graph])[0]
    
    def predict_graphs(
        self,
        graphs: list[dict],
        batch_size: int = 1000,
    ) -> np.ndarray:
        """
        Predict on a list of GraphDicts.
        Returns np.ndarray of shape (B, O).
        Same interface as Megan.predict_graphs().
        """
        all_predictions = []
        
        for start in range(0, len(graphs), batch_size):
            batch_graphs = graphs[start:start + batch_size]
            node_attr, edge_attr, src, dst, batch_idx, num_graphs = (
                self._batch_graphs(batch_graphs)
            )
            
            predictions = self._model.predict(
                node_attr, edge_attr, src, dst, batch_idx, num_graphs,
            )
            all_predictions.append(predictions)
        
        return np.concatenate(all_predictions, axis=0)
    
    def predict_graph(self, graph: dict) -> np.ndarray:
        """Single graph prediction. Returns np.ndarray of shape (O,)."""
        return self.predict_graphs([graph])[0]
    
    @staticmethod
    def _batch_graphs(graphs: list[dict]):
        """Merge multiple GraphDicts into batched arrays (same as PyG batching)."""
        all_nodes = []
        all_edges = []
        all_src = []
        all_dst = []
        all_batch = []
        node_offset = 0
        
        for i, g in enumerate(graphs):
            n = g['node_attributes'].shape[0]
            all_nodes.append(g['node_attributes'])
            all_edges.append(g['edge_attributes'])
            
            ei = g['edge_indices']  # (E, 2)
            all_src.append(ei[:, 0] + node_offset)
            all_dst.append(ei[:, 1] + node_offset)
            all_batch.append(np.full(n, i, dtype=np.uint32))
            
            node_offset += n
        
        return (
            np.concatenate(all_nodes).astype(np.float32),
            np.concatenate(all_edges).astype(np.float32),
            np.concatenate(all_src).astype(np.uint32),
            np.concatenate(all_dst).astype(np.uint32),
            np.concatenate(all_batch).astype(np.uint32),
            len(graphs),
        )
    
    @staticmethod
    def _unbatch_results(batched: dict, graphs: list[dict]) -> list[dict]:
        """Split batched result arrays back into per-graph dicts."""
        results = []
        node_offset = 0
        edge_offset = 0
        
        for i, g in enumerate(graphs):
            num_nodes = g['node_attributes'].shape[0]
            num_edges = g['edge_indices'].shape[0]
            
            results.append({
                'graph_output': batched['graph_output'][i],
                'graph_embedding': batched['graph_embedding'][i],
                'node_importance': batched['node_importance'][node_offset:node_offset+num_nodes],
                'node_importance_norm': batched['node_importance_norm'][node_offset:node_offset+num_nodes],
                'edge_importance': batched['edge_importance'][edge_offset:edge_offset+num_edges],
                'edge_importance_norm': batched['edge_importance_norm'][edge_offset:edge_offset+num_edges],
            })
            
            node_offset += num_nodes
            edge_offset += num_edges
        
        return results
```

### 8. Verification

Before trusting the compiled model, verify numerical equivalence against PyTorch. Compare **intermediate outputs** per layer, not just final predictions — a compensating error between layers could mask a bug.

```python
def verify_compiled_model(
    original: "Megan",
    compiled: "CompiledModel",
    test_graphs: list[dict],
    rtol: float = 1e-4,
    atol: float = 1e-5,
):
    """
    Run both models on the same inputs and assert all outputs match
    within floating-point tolerance.
    """
    original.eval()
    
    for i, g in enumerate(test_graphs):
        torch_info = original.forward_graph(g)
        rust_info = compiled.forward_graph(g)
        
        for key in ['graph_output', 'node_importance', 'node_importance_norm',
                     'edge_importance', 'edge_importance_norm']:
            torch_val = torch_info[key]
            rust_val = rust_info[key]
            
            max_diff = np.max(np.abs(torch_val - rust_val))
            if max_diff > atol:
                print(f"Graph {i}, key '{key}': max diff = {max_diff:.2e}")
            
            np.testing.assert_allclose(
                rust_val, torch_val, rtol=rtol, atol=atol,
                err_msg=f"Mismatch in graph {i}, key '{key}'"
            )
    
    print(f"Verification passed on {len(test_graphs)} graphs!")


def verify_edge_cases(compiled: "CompiledModel", original: "Megan"):
    """Test edge cases that are easy to get wrong."""
    
    # 1. Single-node graph (no edges)
    g_single = {
        'node_attributes': np.random.randn(1, 9).astype(np.float32),
        'edge_attributes': np.zeros((0, 4), dtype=np.float32),
        'edge_indices': np.zeros((0, 2), dtype=np.int64),
    }
    
    # 2. Graph with inf in node features
    g_inf = {
        'node_attributes': np.array([[np.inf, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32),
        'edge_attributes': np.random.randn(2, 4).astype(np.float32),
        'edge_indices': np.array([[0, 1], [1, 0]], dtype=np.int64),
    }
    
    # 3. Two-node graph (minimal edges)
    g_minimal = {
        'node_attributes': np.random.randn(2, 9).astype(np.float32),
        'edge_attributes': np.random.randn(2, 4).astype(np.float32),
        'edge_indices': np.array([[0, 1], [1, 0]], dtype=np.int64),
    }
    
    for name, g in [("single-node", g_single), ("inf-features", g_inf), 
                     ("minimal", g_minimal)]:
        try:
            torch_pred = original.predict_graph(g)
            rust_pred = compiled.predict_graph(g)
            max_diff = np.max(np.abs(torch_pred - rust_pred))
            print(f"  {name}: max diff = {max_diff:.2e} ✓")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
```

Note: small numerical differences (1e-5 to 1e-6) are expected because:
- BN folding changes the order of operations slightly
- `matrixmultiply` may use different accumulation order than PyTorch's BLAS backend
- Projection LayerNorm computed in a single pass vs PyTorch's two-pass algorithm

Run verification on 1000+ molecules to catch rare outliers from accumulated error across layers.

## Implementation Roadmap

### Milestone 1: Hand-Written Single-Layer Proof of Concept

**Goal**: Validate the approach works and measure actual speedup before building the full system.

**Tasks**:
1. Create a new Rust crate with `ndarray`, `matrixmultiply`, and `pyo3` dependencies (no `ndarray-linalg`)
2. Implement the core ops: `linear`, `linear_leaky_relu`, `neighbor_sum`, `gather_concat_with_neighbors`, `scatter_sum`, `scatter_max`, `sigmoid_inplace`
3. Hand-write a single `GraphAttentionLayerV2` forward pass in Rust with hardcoded dimensions (in_dim=64, out_dim=64, hidden_dim=128, edge_dim=4)
4. Write Python code to extract and serialize weights from one trained attention layer (including BN folding)
5. Load weights in Rust, run the layer, compare output against PyTorch for 10 test molecules
6. Benchmark: time the Rust layer vs the PyTorch layer for single-molecule and batch inference

**Expected outcome**: Numerical match within 1e-4, measurable speedup on single-molecule inference. This validates the BN folding math, the `matrixmultiply`-based matmul, and the scatter operations.

### Milestone 2: Complete MEGAN Forward Pass (Hard-Coded)

**Goal**: Full MEGAN inference in Rust, still with dimensions hard-coded for one specific model configuration.

**Tasks**:
1. Implement the full forward pass: embedding (plain linear, no activation) → 3 encoder layers × 2 channels → importance computation with configurable aggregation → node/edge importance normalization → weighted pooling → projection (linear + SiLU + LayerNorm) → final MLP
2. Implement the weight serialization format (with magic + version + CRC32) and Rust-side loader
3. Implement BN folding in Python and verify fused weights produce same output
4. Add the `CompiledModel` Python wrapper with all four methods: `predict_graph`, `predict_graphs`, `forward_graph`, `forward_graphs`
5. Run the verification suite including edge cases (zero-edge graphs, single-node graphs, inf-in-features)
6. Run verification on 1000+ molecules comparing all output keys
7. Benchmark end-to-end: single molecule, batch=32, batch=1000, compare against PyTorch

**Expected outcome**: Full numerical equivalence, concrete speedup numbers to validate estimates.

### Milestone 3: Code Generation

**Goal**: Generalize from the hard-coded MEGAN to any model configuration. Validates the architectural risk of templated code generation before investing in parallelism.

**Tasks**:
1. Write Jinja2 templates for the Rust forward pass, parameterized by: number of layers, units per layer, hidden_dim, num_channels, attention_aggregation, projection_units, final_units, prediction_mode, normalize_embedding, output_norm
2. Templates must emit `// [template: section_name]` comments for error traceability
3. Templates must emit `const` declarations for dimensions (e.g., `const EMBED_DIM: usize = 64;`)
4. Implement `extract_architecture()` and `compile_gnn()` end-to-end, including pre-flight toolchain check
5. Template output snapshot tests: render for 2-3 canonical configs, snapshot the generated Rust source
6. Test with at least 3 different model configurations (varying layers, units, channels)
7. Add support for classification models (output_norm, different final layer handling)

**Expected outcome**: `compile_gnn(model, output_dir)` produces a working compiled model for any MEGAN V2 configuration.

### Milestone 4: Per-Molecule Parallelism with Rayon

**Goal**: Parallel processing of independent molecules for high-throughput batch inference.

**Tasks**:
1. Implement `split_batch()` to partition batched inputs into per-molecule data
2. Pre-allocate one `Scratch` per rayon thread, indexed by `rayon::current_thread_index()`
3. Add rayon `par_iter` over molecules, each using its thread's scratch buffer
4. Benchmark scaling: 1 thread vs 2, 4, 8 threads on batch=1000
5. Compare against PyG batch processing

**Expected outcome**: Near-linear scaling with thread count for batch processing.

### Milestone 5: SMILES-to-Prediction Pipeline (Optional, Deferred)

**Goal**: Accept SMILES strings directly in Rust, eliminating the Python graph conversion overhead.

**Tasks**:
1. Add SMILES parsing in Rust (using a crate like `purr` or `chem-rs`, or port the featurization)
2. Implement the `MoleculeProcessing` featurization logic in Rust
3. Add a `predict_smiles(smiles: Vec<String>) -> Vec<f32>` interface
4. Benchmark the full SMILES-to-prediction pipeline

**Note**: This is a large effort — reimplementing `MoleculeProcessing` in Rust has its own testing burden and the Python-side graph conversion is typically not the bottleneck. Defer unless profiling shows it matters.

## Explicit Out-of-Scope

To prevent scope creep, the following are **not** part of this project:

- **Layer variants V1 and V3**: Start with V2 only (the default). Add V1/V3 only when a concrete need arises.
- **Quantization / int8 inference**: Tempting optimization, massive complexity.
- **Training / backpropagation**: This is inference only.
- **GPU compilation**: CUDA/ROCm codegen is an entirely different project.
- **MVE (Mean Variance Estimation)**: Rarely used optional feature. Can be added later.
- **Dynamic shapes / runtime-configurable architectures**: The generated code is specialized to one model configuration.
- **MeganEnsemble support**: Would require wrapping multiple compiled models. Straightforward to add later but not in initial scope.

## Key Technical Decisions

### f32 vs f64
Use **f32** throughout. PyTorch models are typically trained in f32, and the numerical precision is sufficient for inference. This halves memory bandwidth and doubles SIMD throughput.

### matrixmultiply instead of BLAS
Use the **`matrixmultiply`** crate instead of linking against OpenBLAS/MKL via `ndarray-linalg`. Reasons:

- **Zero system dependencies**: `matrixmultiply` is pure Rust with hand-tuned AVX2/AVX-512 kernels. No libgfortran, no libopenblas-dev, no feature flag gymnastics.
- **Lower dispatch overhead**: For the small matrices in molecular GNNs (~30 rows × 128 cols), full BLAS has significant per-call overhead (thread pool wakeup, workspace allocation, dispatch tables). `matrixmultiply` has near-zero overhead.
- **No thread pool conflicts**: BLAS implementations spawn their own threads, which conflicts with rayon's thread pool causing oversubscription. `matrixmultiply` is single-threaded per call — parallelism comes from rayon at the molecule level.
- **Simpler builds**: Eliminating the BLAS dependency removes the most common source of build failures in Rust numerical code.

For very large matrices (1000+ rows), BLAS with MKL would be faster. But for molecular GNNs this threshold is never reached.

### Per-molecule parallelism
For small molecular graphs, **per-molecule processing with rayon** outperforms batched processing:
- Each molecule can use perfectly-sized scratch buffer slices (no padding)
- Cache locality is better when all data for one molecule fits in L1/L2
- No scatter operations needed across molecule boundaries
- Thread-level parallelism replaces batch-level parallelism
- No BLAS thread pool conflicts since `matrixmultiply` is single-threaded

Pre-allocate one `Scratch` per rayon thread indexed by `rayon::current_thread_index()` to avoid `thread_local!` overhead (OS TLS + RefCell).

### CSR vs COO edge representation
PyG uses COO format (two arrays: src, dst). For the scatter operations in Rust, **CSR format** (compressed sparse row: a `row_ptr` array giving the start/end of each node's edges) enables per-node iteration without branch prediction misses. Convert COO → CSR once at the start of `forward_single`. This is especially important for `scatter_softmax` if ever needed.

### Weight loading: embedded vs file
Start with **file-based** weight loading (the `weights.bin` approach). For deployment, you could optionally embed weights directly into the binary via `include_bytes!()`, eliminating the file dependency entirely — but this makes the binary large and the compile slow, so only do it for final deployment artifacts.

### Build flags
Always compile with `RUSTFLAGS="-C target-cpu=native"`. This enables AVX2/AVX-512 codegen which can double throughput on element-wise ops (sigmoid, activation functions). Also set `lto = "thin"` and `codegen-units = 1` in the generated `Cargo.toml` `[profile.release]` for maximum optimization.
