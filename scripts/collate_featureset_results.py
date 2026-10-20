"""
Collates the results of the AqSolDB feature-set comparison into one report.

Reads every run directory produced by the three arm experiments and reports, per arm,
the prediction metrics and the post-training explanation diagnostic, with the spread
across seeds - because a single run cannot distinguish a real effect from seed noise.

Usage::

    python scripts/collate_featureset_results.py
"""
import glob
import json
import os
import statistics
import typing as t

RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'graph_attention_student', 'experiments', 'results')

ARMS = [
    ('full', 'full (45 cols)', 'everything, including the 5 unportable descriptors'),
    ('portable', 'portable (38 cols)', 'drops crippen/tpsa/lasa/gasteiger/estate'),
    ('portable_strict', 'strict (32 cols)', 'also drops hybridization'),
    ('portable_kekulized', 'portable, kekulized',
     'same 38 cols, aromatic rings kekulized and flags cleared'),
]


def collect(arm: str) -> t.List[dict]:
    """Gathers one record per completed run of ``arm``."""
    pattern = os.path.join(
        RESULTS, f'vgd_torch__megan__aqsoldb__featureset__{arm}', '*', 'experiment_data.json')

    records = []
    for path in sorted(glob.glob(pattern)):
        run_dir = os.path.dirname(path)
        # Skip the fixed "debug" folder (smoke tests) and anything archived aside, so a
        # short test run cannot quietly land in the averages next to a real one.
        if os.path.basename(run_dir) in ('debug',) or '/pilot/' in run_dir:
            continue
        try:
            with open(path) as file:
                data = json.load(file)
        except Exception:
            continue

        record = {
            'dir': os.path.basename(run_dir),
            'num_features': data.get('feature_set', {}).get('num_features'),
        }

        # Epoch count, so runs trained for different lengths are never averaged together.
        history = data.get('margin_met')
        record['epochs'] = len(history) if isinstance(history, list) else None

        metrics = data.get('metrics', {})
        for key in ('r2', 'mae'):
            values = metrics.get(key, {})
            if values:
                record[key] = list(values.values())[0]

        margin = data.get('margin_met')
        if isinstance(margin, list) and margin:
            record['margin_met_final'] = margin[-1]

        # The explanation side comes from the post-training diagnostic.
        report_path = os.path.join(run_dir, 'report.json')
        if os.path.exists(report_path):
            try:
                with open(report_path) as file:
                    report = json.load(file)
                axes = report.get('axes', {})
                record['verdict'] = report.get('overall', {}).get('verdict')
                record['rmse'] = axes.get('prediction', {}).get('rmse')
                record['expl_auc'] = axes.get('explanation_accuracy', {}).get('mean_auc')
                record['expl_verdict'] = axes.get('explanation_accuracy', {}).get('verdict')
                record['fid_sign'] = axes.get('fidelity_sign', {}).get('verdict')
                sparsity = axes.get('coverage', {})
                record['coverage_verdict'] = sparsity.get('verdict')
            except Exception:
                pass

        if 'r2' in record:
            records.append(record)

    # If the arm contains runs of differing length, keep only the longest cohort - the
    # short ones are pilots or smoke tests, not part of the comparison.
    lengths = {r.get('epochs') for r in records if r.get('epochs')}
    if len(lengths) > 1:
        keep = max(lengths)
        dropped = [r for r in records if r.get('epochs') != keep]
        records = [r for r in records if r.get('epochs') == keep]
        print(f'   (ignoring {len(dropped)} run(s) with fewer epochs than {keep})')

    return records


def summarize(values: t.List[float]) -> str:
    """mean +/- spread, or a single value when there is only one run."""
    clean = [v for v in values if v is not None]
    if not clean:
        return '     -    '
    if len(clean) == 1:
        return f'{clean[0]:.4f}     '
    return f'{statistics.mean(clean):.4f} ±{statistics.stdev(clean):.4f}'


def main() -> int:
    print('=' * 92)
    print('AqSolDB feature-set comparison - what the browser-portable featurizer costs')
    print('=' * 92)

    table = {}
    for arm, label, note in ARMS:
        records = collect(arm)
        table[arm] = records
        print(f'\n{label:<22} {note}')
        if not records:
            print('   (no completed runs)')
            continue
        print(f'   {"run":<26} {"cols":>5} {"R2":>8} {"RMSE":>8} {"MAE":>8} '
              f'{"expl AUC":>9} {"margin":>7}  verdict')
        for record in records:
            print(f'   {record["dir"][:26]:<26} {record.get("num_features", "?"):>5} '
                  f'{record.get("r2", float("nan")):>8.4f} '
                  f'{record.get("rmse", float("nan")):>8.4f} '
                  f'{record.get("mae", float("nan")):>8.4f} '
                  f'{record.get("expl_auc", float("nan")):>9.4f} '
                  f'{record.get("margin_met_final", float("nan")):>7.3f}  '
                  f'{record.get("verdict", "-")}')

    print('\n' + '=' * 92)
    print('SUMMARY (mean ± stdev across seeds)')
    print('=' * 92)
    print(f'{"arm":<22} {"n":>2} {"cols":>5} {"test R2":>16} {"RMSE":>16} {"expl AUC":>16}')
    for arm, label, _ in ARMS:
        records = table.get(arm, [])
        if not records:
            continue
        print(f'{label:<22} {len(records):>2} {records[0].get("num_features", "?"):>5} '
              f'{summarize([r.get("r2") for r in records]):>16} '
              f'{summarize([r.get("rmse") for r in records]):>16} '
              f'{summarize([r.get("expl_auc") for r in records]):>16}')

    # The headline number: what does restricting the featurizer actually cost?
    full = [r.get('r2') for r in table.get('full', []) if r.get('r2') is not None]
    if full:
        base = statistics.mean(full)
        print()
        for arm, label, _ in ARMS[1:]:
            values = [r.get('r2') for r in table.get(arm, []) if r.get('r2') is not None]
            if not values:
                continue
            delta = statistics.mean(values) - base

            # With one run per arm there is no spread to compare against, and calling a
            # difference significant off a zero spread would be an artifact rather than a
            # finding. Say so instead of pretending to a verdict.
            if len(full) < 2 or len(values) < 2:
                print(f'  {label:<22} ΔR² vs full = {delta:+.4f}   '
                      f'(n={len(values)} vs {len(full)} - too few runs to separate this '
                      f'from seed noise)')
                continue

            # Pooled standard deviation across both arms, so the comparison accounts for
            # the variability of each rather than only the reference.
            spread = ((statistics.stdev(full) ** 2 + statistics.stdev(values) ** 2) / 2) ** 0.5
            verdict = ('within seed noise' if abs(delta) <= spread
                       else 'exceeds seed noise')
            print(f'  {label:<22} ΔR² vs full = {delta:+.4f}   '
                  f'(pooled seed spread ±{spread:.4f}) -> {verdict}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
