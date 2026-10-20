"""
Speed comparison between the PyTorch/PyG model and its ONNX export.

Measures the forward pass only. Featurization is excluded from both sides - it is
identical work either way, and the point here is the inference cost. The PyG collation
into a batch is likewise done once, outside the timing loop, for both.

Both runtimes are pinned to the same thread count so that the comparison is not
accidentally a measurement of how many cores each one decided to use.

Usage::

    python scripts/benchmark_onnx.py                 # CPU, 1 thread and all threads
    python scripts/benchmark_onnx.py --device cuda   # add the GPU comparison
    python scripts/benchmark_onnx.py --num-molecules 2000
"""
import argparse
import glob
import json
import os
import statistics
import tempfile
import time
import typing as t

import numpy as np
import torch

AQSOLDB = os.path.expanduser('~/.visual_graph_datasets/datasets/aqsoldb')

#: Batch sizes to sweep. 1 is the interactive case a static site actually hits; the
#: larger ones are the screening case.
BATCH_SIZES = (1, 8, 32, 128, 512)


def load_smiles(limit: int) -> t.List[str]:
    """
    Reads SMILES from the local aqsoldb visual graph dataset, falling back to a small
    built-in list when it is not present.
    """
    paths = sorted(glob.glob(os.path.join(AQSOLDB, '*.json')))[:limit]
    smiles = []
    for path in paths:
        try:
            with open(path) as file:
                smiles.append(json.load(file)['smiles'])
        except Exception:
            continue

    if not smiles:
        print(f'! {AQSOLDB} not found, falling back to a handful of built-in molecules')
        smiles = ['CCO', 'c1ccccc1O', 'CC(=O)Oc1ccccc1C(=O)O',
                  'CN1C=NC2=C1C(=O)N(C)C(=O)N2C', 'FC(F)(F)c1ccccc1'] * 40

    return smiles


def build_batches(graphs, batch_sizes, num_channels):
    """
    Pre-collates one batch per requested size, so that neither runtime pays for the
    collation inside the timing loop.
    """
    from torch_geometric.loader import DataLoader
    from graph_attention_student.torch.data import data_list_from_graphs

    batches = {}
    for size in batch_sizes:
        if size > len(graphs):
            continue
        subset = graphs[:size]
        data = next(iter(DataLoader(
            data_list_from_graphs(subset), batch_size=size, shuffle=False)))
        num_graphs = int(data.batch.max()) + 1
        args = (
            data.x, data.edge_index, data.edge_attr, data.batch,
            torch.arange(num_graphs, dtype=torch.long),
            torch.ones(data.x.shape[0], num_channels),
        )
        batches[size] = (data, args)

    return batches


def time_it(function, repeats: int, warmup: int, sync=None) -> t.Tuple[float, float]:
    """
    :returns: (median, stdev) wall-clock seconds per call.
    """
    for _ in range(warmup):
        function()
    if sync:
        sync()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        if sync:
            sync()
        samples.append(time.perf_counter() - start)

    return statistics.median(samples), (statistics.stdev(samples) if len(samples) > 1 else 0.0)


def run_cpu(model, model_path, batches, threads: int, repeats: int, warmup: int):
    """Benchmarks PyTorch and ONNX Runtime on the CPU at a fixed thread count."""
    import onnxruntime as ort
    from graph_attention_student.torch.export.wrapper import INPUT_NAMES

    torch.set_num_threads(threads)

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model_path, options, providers=['CPUExecutionProvider'])

    model = model.to('cpu').eval()
    rows = []
    for size, (data, args) in batches.items():
        feeds = {name: tensor.numpy() for name, tensor in zip(INPUT_NAMES, args)}

        def torch_call():
            with torch.no_grad():
                model(data)

        def onnx_call():
            session.run(None, feeds)

        torch_median, torch_std = time_it(torch_call, repeats, warmup)
        onnx_median, onnx_std = time_it(onnx_call, repeats, warmup)

        rows.append({
            'batch': size,
            'torch_ms': torch_median * 1e3,
            'torch_std': torch_std * 1e3,
            'onnx_ms': onnx_median * 1e3,
            'onnx_std': onnx_std * 1e3,
            'speedup': torch_median / onnx_median,
            'torch_mps': size / torch_median,
            'onnx_mps': size / onnx_median,
        })

    return rows


def run_cuda(model, model_path, batches, repeats: int, warmup: int,
             ort_gpu_results: t.Optional[dict] = None):
    """
    Benchmarks PyTorch on the GPU, and ONNX Runtime on the GPU when a CUDA-capable
    build is installed.

    PyTorch is measured twice: with the batch already resident on the device, and
    including the host-to-device transfer. The second is the one comparable to the ONNX
    Runtime figures, which are fed from host memory and therefore pay for the copy.

    :param ort_gpu_results: optionally, ``{batch_size: seconds}`` measured elsewhere.
        The stock onnxruntime wheel is CPU-only, so the CUDA numbers usually have to be
        produced by a separate onnxruntime-gpu install and passed in here.
    """
    import onnxruntime as ort
    from graph_attention_student.torch.export.wrapper import INPUT_NAMES

    model = model.to('cuda').eval()
    has_cuda_ep = 'CUDAExecutionProvider' in ort.get_available_providers()

    session = None
    if has_cuda_ep:
        session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        assert 'CUDAExecutionProvider' in session.get_providers(), \
            'the session silently fell back to the CPU provider'

    rows = []
    for size, (data, args) in batches.items():
        data_cuda = data.to('cuda')

        def torch_call():
            with torch.no_grad():
                model(data_cuda)

        def torch_call_with_transfer():
            with torch.no_grad():
                model(data.to('cuda'))

        torch_median, torch_std = time_it(
            torch_call, repeats, warmup, sync=torch.cuda.synchronize)
        transfer_median, _ = time_it(
            torch_call_with_transfer, repeats, warmup, sync=torch.cuda.synchronize)

        row = {
            'batch': size,
            'torch_ms': torch_median * 1e3,
            'torch_std': torch_std * 1e3,
            'torch_mps': size / torch_median,
            'torch_transfer_ms': transfer_median * 1e3,
            'torch_transfer_mps': size / transfer_median,
        }

        onnx_median = None
        if session is not None:
            feeds = {name: tensor.numpy() for name, tensor in zip(INPUT_NAMES, args)}

            def onnx_call():
                session.run(None, feeds)

            onnx_median, _ = time_it(onnx_call, repeats, warmup)
        elif ort_gpu_results and str(size) in ort_gpu_results:
            onnx_median = ort_gpu_results[str(size)]

        if onnx_median is not None:
            row.update({
                'onnx_ms': onnx_median * 1e3,
                'speedup': transfer_median / onnx_median,
                'onnx_mps': size / onnx_median,
            })

        rows.append(row)

    return rows, any('onnx_ms' in row for row in rows)


def print_table(title: str, rows: t.List[dict], has_onnx: bool = True) -> None:
    print(f'\n{title}')
    transfer = any('torch_transfer_ms' in row for row in rows)
    if has_onnx and transfer:
        print(f'  {"batch":>6} {"torch resident":>16} {"torch +copy":>14} '
              f'{"ONNX Runtime":>14} {"speedup":>9} {"onnx mol/s":>12}')
        for row in rows:
            print(f'  {row["batch"]:>6} {row["torch_ms"]:>12.3f} ms '
                  f'{row["torch_transfer_ms"]:>10.3f} ms '
                  f'{row["onnx_ms"]:>10.3f} ms '
                  f'{row["speedup"]:>7.2f}x {row["onnx_mps"]:>12.0f}')
    elif has_onnx:
        print(f'  {"batch":>6} {"PyTorch/PyG":>18} {"ONNX Runtime":>18} '
              f'{"speedup":>9} {"torch mol/s":>12} {"onnx mol/s":>12}')
        for row in rows:
            print(f'  {row["batch"]:>6} '
                  f'{row["torch_ms"]:>11.3f} ms      '
                  f'{row["onnx_ms"]:>10.3f} ms      '
                  f'{row["speedup"]:>7.2f}x '
                  f'{row["torch_mps"]:>12.0f} {row["onnx_mps"]:>12.0f}')
    else:
        print(f'  {"batch":>6} {"torch resident":>16} {"torch +copy":>14} '
              f'{"torch mol/s":>12}')
        for row in rows:
            print(f'  {row["batch"]:>6} {row["torch_ms"]:>12.3f} ms '
                  f'{row.get("torch_transfer_ms", float("nan")):>10.3f} ms '
                  f'{row["torch_mps"]:>12.0f}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-molecules', type=int, default=600)
    parser.add_argument('--repeats', type=int, default=25)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'both'])
    parser.add_argument('--units', default='64,64,64',
                        help='encoder units, matching the CLI default')
    parser.add_argument('--model-path', default=None,
                        help='reuse an already exported model.onnx instead of exporting')
    parser.add_argument('--ort-gpu-results', default=None,
                        help='JSON of {batch_size: seconds} from an onnxruntime-gpu run, '
                             'since the stock onnxruntime wheel is CPU-only')
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings('ignore')

    from visual_graph_datasets.processing.molecules import MoleculeProcessing
    from graph_attention_student.torch.megan import Megan
    from graph_attention_student.torch.export import export_megan_onnx

    processing = MoleculeProcessing()
    smiles = load_smiles(args.num_molecules)

    graphs = []
    for value in smiles:
        try:
            graph = processing.process(value)
            # Megan cannot process an edgeless graph; skip rather than crash.
            if np.asarray(graph['edge_indices']).ndim == 2:
                graphs.append(graph)
        except Exception:
            continue

    units = [int(v) for v in args.units.split(',')]
    torch.manual_seed(3)
    model = Megan(
        node_dim=processing.get_num_node_attributes(),
        edge_dim=processing.get_num_edge_attributes(),
        units=units,
        num_channels=2,
        projection_units=[64],
        final_units=[64, 32, 1],
    )
    model.eval()

    sizes = [s for s in BATCH_SIZES if s <= len(graphs)]
    batches = build_batches(graphs, sizes, model.num_channels)

    if args.model_path:
        model_path = args.model_path
    else:
        directory = tempfile.mkdtemp()
        model_path = export_megan_onnx(
            model, directory, graphs[:8], processing=processing)['model']

    node_counts = [batches[s][0].x.shape[0] for s in sizes]
    print('=' * 78)
    print('MEGAN inference: PyTorch/PyG vs ONNX Runtime')
    print('=' * 78)
    print(f'molecules      : {len(graphs)} from aqsoldb '
          f'(mean {np.mean([g["node_attributes"].shape[0] for g in graphs]):.1f} atoms)')
    print(f'model          : units={units} channels={model.num_channels} '
          f'final=[64, 32, 1]')
    print(f'onnx model     : {os.path.getsize(model_path) / 1024:.0f} KB')
    print(f'batch sizes    : {sizes}  (nodes: {node_counts})')
    print(f'timing         : median of {args.repeats} runs after {args.warmup} warmup')

    if args.device in ('cpu', 'both'):
        rows = run_cpu(model, model_path, batches, 1, args.repeats, args.warmup)
        print_table('CPU, 1 thread', rows)

        threads = min(8, os.cpu_count() or 1)
        rows = run_cpu(model, model_path, batches, threads, args.repeats, args.warmup)
        print_table(f'CPU, {threads} threads', rows)

    if args.device in ('cuda', 'both'):
        if not torch.cuda.is_available():
            print('\nCUDA is not available; skipping the GPU comparison.')
        else:
            ort_gpu_results = None
            if args.ort_gpu_results and os.path.exists(args.ort_gpu_results):
                ort_gpu_results = json.load(open(args.ort_gpu_results))

            rows, has_onnx = run_cuda(
                model, model_path, batches, args.repeats, args.warmup, ort_gpu_results)
            title = f'GPU ({torch.cuda.get_device_name(0)})'
            if not has_onnx:
                title += ' - no onnxruntime CUDA numbers available'
            print_table(title, rows, has_onnx=has_onnx)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
