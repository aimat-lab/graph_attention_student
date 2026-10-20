"""
Statistical analysis of the AqSolDB feature-set comparison.

The three arms share a pinned train/val/test split per seed - the same seed means the
same molecules in every arm - so the arms are *paired*. A paired test is both the correct
analysis and a much more sensitive one than treating the arms as independent samples,
because it removes the split-to-split variance that dominates the raw spread.

Usage::

    python scripts/analyze_featureset_results.py
"""
import glob
import json
import os
import typing as t

import numpy as np
from scipy import stats

RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'graph_attention_student', 'experiments', 'results')

ARMS = [
    ('full', 'full (45 cols)'),
    ('portable', 'portable (38 cols)'),
    ('portable_strict', 'strict (32 cols)'),
    ('portable_kekulized', 'portable kekulized'),
]

METRICS = [
    ('r2', 'test R2', True),
    ('rmse', 'test RMSE', False),
    ('expl_auc', 'explanation AUC', True),
]


def collect(arm: str) -> t.Dict[int, dict]:
    """:returns: ``{split_seed: metrics}`` for every completed run of ``arm``."""
    pattern = os.path.join(
        RESULTS, f'vgd_torch__megan__aqsoldb__featureset__{arm}', '*', 'experiment_data.json')

    by_seed = {}
    for path in sorted(glob.glob(pattern)):
        run_dir = os.path.dirname(path)
        if os.path.basename(run_dir) == 'debug' or '/pilot/' in run_dir:
            continue

        try:
            data = json.load(open(path))
        except Exception:
            continue

        # Only fully trained runs; anything shorter is a pilot or a smoke test.
        if len(data.get('margin_met') or []) != 150:
            continue

        seed = None
        try:
            meta = json.load(open(os.path.join(run_dir, 'experiment_meta.json')))
            seed = meta['parameters']['SPLIT_SEED']['value']
        except Exception:
            continue

        record = {}
        metrics = data.get('metrics', {})
        for key in ('r2', 'mae'):
            values = metrics.get(key, {})
            if values:
                record[key] = list(values.values())[0]

        report_path = os.path.join(run_dir, 'report.json')
        if os.path.exists(report_path):
            report = json.load(open(report_path))
            axes = report.get('axes', {})
            record['rmse'] = axes.get('prediction', {}).get('rmse')
            record['expl_auc'] = axes.get('explanation_accuracy', {}).get('mean_auc')

        if 'r2' in record and seed is not None:
            by_seed[seed] = record

    return by_seed


def paired_test(baseline: t.List[float], other: t.List[float]) -> dict:
    """Paired t-test plus the effect size, on matched-seed pairs."""
    diff = np.array(other) - np.array(baseline)
    result = {
        'mean_diff': float(diff.mean()),
        'sd_diff': float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
        'n': len(diff),
    }
    if len(diff) > 1 and diff.std(ddof=1) > 0:
        statistic, pvalue = stats.ttest_rel(other, baseline)
        result['t'] = float(statistic)
        result['p'] = float(pvalue)
        result['cohens_d'] = float(diff.mean() / diff.std(ddof=1))
        # 95% CI on the mean paired difference.
        margin = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
        result['ci'] = (float(diff.mean() - margin), float(diff.mean() + margin))
    return result


def main() -> int:
    data = {arm: collect(arm) for arm, _ in ARMS}

    print('=' * 88)
    print('AqSolDB: what the browser-portable featurizer costs')
    print('=' * 88)
    for arm, label in ARMS:
        got = sorted(data[arm].keys())
        print(f'  {label:<24} n={len(got)}  seeds {got}')
    print('  (same seed = same train/val/test split in every arm)')

    for key, label, higher_better in METRICS:
        print(f'\n--- {label} {"(higher better)" if higher_better else "(lower better)"} ---')
        all_seeds = sorted(set().union(*[set(v.keys()) for v in data.values()]))
        print(f'  {"seed":<6}' + ''.join(f'{lbl:>22}' for _, lbl in ARMS))
        for seed in all_seeds:
            row = f'  {seed:<6}'
            for arm, _ in ARMS:
                value = data[arm].get(seed, {}).get(key)
                row += f'{value:>22.4f}' if value is not None else f'{"-":>22}'
            print(row)

        row = f'  {"mean":<6}'
        for arm, _ in ARMS:
            clean = [r.get(key) for r in data[arm].values() if r.get(key) is not None]
            if len(clean) > 1:
                row += f'{np.mean(clean):>15.4f} ±{np.std(clean, ddof=1):.4f}'
            elif clean:
                row += f'{np.mean(clean):>15.4f}       '
            else:
                row += f'{"-":>22}'
        print(row)

        # Each arm against the baseline it is actually asking a question about: the
        # reduced feature sets against "full", but the kekulized arm against "portable",
        # since it holds the columns fixed and changes only the aromatic representation.
        comparisons = [
            ('full', 'portable', 'portable vs full'),
            ('full', 'portable_strict', 'strict vs full'),
            ('portable', 'portable_kekulized', 'kekulized vs portable'),
        ]

        print()
        for base_arm, other_arm, label_cmp in comparisons:
            # Each comparison is paired on the seeds those two arms actually share, so a
            # partially finished arm never shrinks an unrelated comparison.
            shared = sorted(set(data[base_arm].keys()) & set(data[other_arm].keys()))
            base = [data[base_arm][s].get(key) for s in shared]
            other = [data[other_arm][s].get(key) for s in shared]
            if any(v is None for v in base + other) or len(shared) < 2:
                continue
            res = paired_test(base, other)
            if 'p' not in res:
                continue
            sig = ('SIGNIFICANT' if res['p'] < 0.05 else 'not significant')
            print(f'  {label_cmp:<24} n={len(shared)}  Δ = {res["mean_diff"]:+.4f}  '
                  f'95% CI [{res["ci"][0]:+.4f}, {res["ci"][1]:+.4f}]  '
                  f'p = {res["p"]:.4f}  d = {res["cohens_d"]:+.2f}  -> {sig}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
