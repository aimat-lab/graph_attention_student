"""
Broad Python-vs-JavaScript featurizer validation.

The fixtures the CI test uses cover 20 hand-picked molecules chosen to stress specific
branches. This writes a much larger sample so the agreement claim rests on real chemistry
at scale rather than on cases that were selected because they are interesting.

    python scripts/validate_js_featurizer.py --limit 1000
    node js/validate_broad.mjs
"""
import argparse
import glob
import json
import os

import numpy as np

from graph_attention_student.torch.export.portable_processing import (
    PortableMoleculeProcessing)
from graph_attention_student.torch.export.featurization import derive_featurization_spec

DATASETS = ['aqsoldb', 'lipophilicity', 'logp']
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'js', 'test', 'broad')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=1000,
                        help='molecules to take from each dataset')
    args = parser.parse_args()

    processing = PortableMoleculeProcessing()
    os.makedirs(OUT, exist_ok=True)

    with open(os.path.join(OUT, 'featurization.json'), 'w') as file:
        json.dump(derive_featurization_spec(processing).spec, file)

    records, skipped = [], 0
    for name in DATASETS:
        paths = sorted(glob.glob(
            os.path.expanduser(f'~/.visual_graph_datasets/datasets/{name}/*.json')))
        for path in paths[:args.limit]:
            try:
                smiles = json.load(open(path))['smiles']
                graph = processing.process(smiles)
            except Exception:
                skipped += 1
                continue
            # Edgeless molecules are not featurizable by either side.
            if np.asarray(graph['edge_indices']).ndim != 2:
                skipped += 1
                continue
            records.append({
                'smiles': smiles,
                'node_attributes': np.asarray(graph['node_attributes']).tolist(),
                'edge_indices': np.asarray(graph['edge_indices']).tolist(),
                'edge_attributes': np.asarray(graph['edge_attributes']).tolist(),
            })

    path = os.path.join(OUT, 'cases.json')
    with open(path, 'w') as file:
        json.dump(records, file)

    values = sum(len(r['node_attributes']) * len(r['node_attributes'][0])
                 + len(r['edge_attributes']) * len(r['edge_attributes'][0])
                 for r in records)
    print(f'wrote {len(records)} molecules ({skipped} skipped) to {path}')
    print(f'{values} feature values for the JavaScript side to reproduce')
    print(f'file size: {os.path.getsize(path) / 1e6:.1f} MB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
