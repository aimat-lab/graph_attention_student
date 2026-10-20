"""
Measures how many MEGAN training runs fit concurrently on one local GPU.

Builds the real aqsoldb model configuration - including the 1.6.0 redesign terms, since
RDT runs a second forward pass and therefore changes both the memory footprint and the
step time - and runs a fixed number of training steps against real aqsoldb graphs.

Run one instance per concurrent slot; each prints a JSON line with its own throughput
and peak memory. Comparing single-slot throughput against N-slot throughput gives the
scaling efficiency.

Usage::

    python scripts/gpu_capacity_probe.py --steps 60 --tag solo
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

AQSOLDB = os.path.expanduser('~/.visual_graph_datasets/datasets/aqsoldb')

#: The aqsoldb model configuration, plus the redesign explanation terms.
MODEL_CONFIG = dict(
    units=[128, 128, 128],
    hidden_units=128,
    importance_units=[],
    projection_units=[64, 128],
    final_units=[128, 64, 1],
    num_channels=2,
    importance_factor=1.0,
    importance_offset=1.0,
    sparsity_factor=1.0,
    fidelity_factor=0.1,
    normalize_embedding=True,
    attention_aggregation='max',
    contrastive_factor=1.0,
    contrastive_noise=0.1,
    contrastive_tau=0.1,
    contrastive_temp=1.0,
    contrastive_beta=1.0,
    prediction_mode='regression',
    importance_mode='regression',
    learning_rate=1e-5,
    # -- 1.6.0 redesign
    explanation_objective='difference_margin',
    difference_margin_scale=4.0,
    regression_margin=0.1,
    rdt_factor=1.0,
    rdt_mask_only=True,
    rdt_use_raw=True,
    rdt_warmup=0.0,
    rdt_ramp=0.0,
    rdt_samples=1,
    polar_factor=0.3,
    spread_factor=0.5,
)


def load_graphs(limit: int):
    """Loads graph dicts straight out of the visual graph dataset json files."""
    paths = sorted(glob.glob(os.path.join(AQSOLDB, '*.json')))[:limit]
    graphs = []
    for path in paths:
        try:
            with open(path) as file:
                element = json.load(file)
            graph = element['graph']
            for key in ('node_attributes', 'edge_attributes', 'edge_indices',
                        'node_indices'):
                graph[key] = np.array(graph[key])
            graph['graph_labels'] = np.array(element['target'], dtype='float32')
            if graph['edge_indices'].ndim == 2 and len(graph['edge_indices']):
                graphs.append(graph)
        except Exception:
            continue
    return graphs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-graphs', type=int, default=1500)
    parser.add_argument('--node-dim', type=int, default=45)
    parser.add_argument('--tag', default='run')
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings('ignore')

    from torch_geometric.loader import DataLoader
    from graph_attention_student.torch.megan import Megan
    from graph_attention_student.torch.data import data_list_from_graphs

    graphs = load_graphs(args.num_graphs)

    # Slice the node features down, exactly as the reduced-featurizer arms will.
    if args.node_dim < 45:
        for graph in graphs:
            graph['node_attributes'] = graph['node_attributes'][:, :args.node_dim]

    data_list = data_list_from_graphs(graphs)
    loader = DataLoader(data_list, batch_size=args.batch_size, shuffle=True)

    import pytorch_lightning as pl

    model = Megan(node_dim=args.node_dim, edge_dim=11, **MODEL_CONFIG)
    torch.cuda.reset_peak_memory_stats()

    # A callback rather than a manual loop: training_step reads self.trainer for the RDT
    # warmup/ramp schedule, so the model has to be attached to a real Trainer.
    class Timer(pl.Callback):
        def __init__(self, warmup):
            self.warmup = warmup
            self.start = None
            self.count = 0

        def on_train_batch_end(self, *a, **kw):
            self.count += 1
            if self.count == self.warmup:
                torch.cuda.synchronize()
                self.start = time.perf_counter()

    timer = Timer(args.warmup)
    trainer = pl.Trainer(
        max_steps=args.steps + args.warmup,
        accelerator='gpu',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[timer],
    )
    trainer.fit(model, train_dataloaders=loader)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - timer.start
    peak = torch.cuda.max_memory_allocated() / 1e9

    print(json.dumps({
        'tag': args.tag,
        'pid': os.getpid(),
        'steps': args.steps,
        'seconds': round(elapsed, 3),
        'steps_per_second': round(args.steps / elapsed, 2),
        'peak_gb': round(peak, 3),
        'node_dim': args.node_dim,
    }), flush=True)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
