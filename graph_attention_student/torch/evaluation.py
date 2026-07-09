"""Post-training diagnostic protocol for MEGAN models.

This module consolidates the scattered per-run evaluation signals into a single,
*agent-facing* verdict about whether a trained MEGAN model produced usable
explanations. It is designed for the scenario where an autonomous agent trains a
model unattended and then needs to decide - without a human looking at PDFs -
whether the run succeeded or should be repeated.

The central entry point is :func:`generate_diagnostic_report`, which writes:

- ``report.json``  - machine-readable: the actual scalar values each verdict is
  based on, PLUS a PASS / WARN / FAIL verdict per axis and an overall verdict.
- ``fidelity.png``  - leave-one-out deviation histograms per channel/target.
- ``examples.png``  - a small grid of random example explanations (the coverage
  judgement is inherently visual, so the agent must actually look).
- ``scorecard.png`` - a one-glance panel pairing explanation-accuracy against
  coverage per channel (they are antagonists) with the verdicts printed on it.

The five diagnostic axes (encoding the maintainer's expert loop):

1. ``prediction``            - R2/MAE or Acc/F1. Context; never hard-fails.
2. ``explanation_accuracy``  - can the explanation subgraph alone solve the proxy
   task (per-channel AUC)? HARD FAIL gate: ~0.5 == dead.
3. ``fidelity_sign``         - does the leave-one-out deviation *distribution* sit
   on the canonically expected side for the majority of graphs?
4. ``fidelity_magnitude``    - do the channels actually move the output when masked?
5. ``coverage``              - are the masks focused (not the whole graph, not
   invisible)? Soft / visual.

All heavy computations reuse the model's own trusted methods
(``forward_graphs``, ``_predict_approximate``, ``leave_one_out_deviations``) and
the existing ``plot_leave_one_out_analysis`` visualization - this module only
adds consolidation, thresholds/verdicts, and the JSON writer.
"""

import os
import json
import typing as t
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, f1_score
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import plot_node_importances_background
from visual_graph_datasets.visualization.importances import plot_edge_importances_background

from graph_attention_student.visualization import plot_leave_one_out_analysis


# == VERDICT THRESHOLDS ==
# All thresholds are module-level constants so they are easy to tune in one place.

# -- explanation accuracy (HARD FAIL gate) --
# Mean per-channel AUC of the explanation proxy task. ~0.5 means the explanation
# subgraph carries no task-relevant information at all.
EXPL_ACC_FAIL: float = 0.6
EXPL_ACC_WARN: float = 0.7

# -- fidelity sign consistency --
# Fraction of graphs whose leave-one-out deviation sits on the canonically
# expected side for that channel. Below 0.5 == the majority is *inverted*.
SIGN_CONSISTENCY_FAIL: float = 0.5
SIGN_CONSISTENCY_WARN: float = 0.7

# -- fidelity magnitude --
# Mean absolute leave-one-out deviation, relative to the target scale. Below this
# the channels barely influence the output at all.
FIDELITY_MAGNITUDE_WARN: float = 0.05

# -- coverage (soft / visual) --
# A node counts as "active" if its per-graph-normalized importance exceeds this.
COVERAGE_ACTIVE_THRESHOLD: float = 0.5
# A channel whose max importance is below this on a graph is "invisible" there.
COVERAGE_EMPTY_EPS: float = 0.1
# WARN if a channel is invisible on more than this fraction of graphs ...
COVERAGE_EMPTY_FRAC_WARN: float = 0.5
# ... or if it activates more than this fraction of nodes on average.
COVERAGE_SATURATED_FRAC_WARN: float = 0.9

# -- prediction (context; soft) --
PRED_R2_WARN: float = 0.3
PRED_ACC_WARN: float = 0.6

# Number of random example graphs shown in examples.png
NUM_EXAMPLES: int = 8


# == VERDICT PLUMBING ==

PASS = 'PASS'
WARN = 'WARN'
FAIL = 'FAIL'
_RANK = {PASS: 0, WARN: 1, FAIL: 2}


def _worst(*verdicts: str) -> str:
    """Return the most severe verdict among the arguments."""
    return max(verdicts, key=lambda v: _RANK[v]) if verdicts else PASS


def _f(value) -> float:
    """Convert to a JSON-friendly python float (NaN/inf -> None handled by caller)."""
    return float(value)


# == RAW GATHERING ==

def _compute_raw(model, graphs: List[dict]) -> dict:
    """
    Run the (relatively expensive) model queries once and return the raw arrays
    that all axes are computed from. Kept separate so images and metrics never
    recompute the leave-one-out deviations.
    """
    results: List[dict] = model.forward_graphs(graphs)
    values_true = np.array([np.asarray(g['graph_labels']).flatten() for g in graphs])
    values_pred = np.array([np.asarray(r['graph_output']).flatten() for r in results])
    # leave_one_out: (B, O, K)
    leave_one_out = model.leave_one_out_deviations(graphs)
    return {
        'results': results,
        'values_true': values_true,
        'values_pred': values_pred,
        'leave_one_out': leave_one_out,
    }


# == INDIVIDUAL AXES ==

def _axis_prediction(model, raw: dict) -> dict:
    values_true = raw['values_true']
    values_pred = raw['values_pred']

    if model.prediction_mode == 'regression':
        r2 = _f(r2_score(values_true, values_pred))
        mae = _f(mean_absolute_error(values_true, values_pred))
        rmse = _f(np.sqrt(np.mean((values_true - values_pred) ** 2)))
        verdict = WARN if r2 < PRED_R2_WARN else PASS
        return {
            'verdict': verdict,
            'r2': r2, 'mae': mae, 'rmse': rmse,
            'threshold': {'r2_warn': PRED_R2_WARN},
        }
    else:
        yt = np.argmax(values_true, axis=1)
        yp = np.argmax(values_pred, axis=1)
        acc = _f(accuracy_score(yt, yp))
        f1 = _f(f1_score(yt, yp, average='macro'))
        verdict = WARN if acc < PRED_ACC_WARN else PASS
        return {
            'verdict': verdict,
            'accuracy': acc, 'f1_macro': f1,
            'threshold': {'accuracy_warn': PRED_ACC_WARN},
        }


def _axis_explanation_accuracy(model, raw: dict) -> dict:
    """
    HARD FAIL gate: per-channel AUC of the explanation proxy task.

    A subtle failure mode is that the metric cannot be *computed* at all - e.g. the
    model was reloaded without its ``importance_mode`` (so ``_predict_approximate``
    never binarizes the target) or the eval set is single-class. In that case the
    axis must NOT silently report ~0.5 as if the explanations were random - it
    reports ``computed: False`` with an explicit ``error`` so the agent can tell a
    genuinely-random model apart from a broken measurement. The verdict is still
    FAIL (safe default), but the reason is diagnostic rather than misleading.
    """
    threshold = {'fail_below': EXPL_ACC_FAIL, 'warn_below': EXPL_ACC_WARN}

    # -- precondition: explanation co-training must be active for the proxy task --
    # Without an importance_mode, _predict_approximate cannot form the binary proxy
    # target and the whole axis is meaningless.
    if model.importance_mode is None:
        return {
            'verdict': FAIL,
            'computed': False,
            'error': ('importance_mode is None on the model - explanation co-training was '
                      'disabled or not restored on load; explanation accuracy is not measurable'),
            'mean_auc': None,
            'per_channel_auc': None,
            'per_channel_opt_acc': None,
            'threshold': threshold,
        }

    results = raw['results']
    values_true = raw['values_true']
    # approx_true: (N, K) binary target per channel; approx_pred: (N, K) in [-1, 1]
    approx_true, approx_pred = model._predict_approximate(
        results=results, values_true=values_true,
    )

    per_channel_auc: List[Optional[float]] = []
    per_channel_opt_acc: List[Optional[float]] = []
    uncomputable: List[int] = []
    for k in range(model.num_channels):
        # channel has no proxy target column (shape mismatch) -> not measurable
        if k >= approx_true.shape[1]:
            per_channel_auc.append(None)
            per_channel_opt_acc.append(None)
            uncomputable.append(k)
            continue

        yt_k = approx_true[:, k].astype(float)
        yp_k = approx_pred[:, k]
        uniq = np.unique(yt_k)

        # The proxy target must be binary {0, 1}. If it is continuous (the classic
        # symptom of a model reloaded without importance_mode) or single-class, AUC
        # is not computable -> record as uncomputable rather than faking 0.5.
        if not np.all(np.isin(uniq, [0.0, 1.0])) or len(uniq) < 2:
            per_channel_auc.append(None)
            per_channel_opt_acc.append(None)
            uncomputable.append(k)
            continue

        try:
            auc_k = float(roc_auc_score(yt_k, yp_k))
        except ValueError:
            per_channel_auc.append(None)
            per_channel_opt_acc.append(None)
            uncomputable.append(k)
            continue
        per_channel_auc.append(round(auc_k, 4))

        best_acc = 0.5
        for th in np.unique(yp_k):
            best_acc = max(best_acc, float(np.mean(yt_k == (yp_k > th).astype(float))))
        per_channel_opt_acc.append(round(best_acc, 4))

    valid = [a for a in per_channel_auc if a is not None]

    # nothing could be measured on any channel -> broken measurement, not random model
    if not valid:
        return {
            'verdict': FAIL,
            'computed': False,
            'error': ('explanation proxy AUC could not be computed on any channel '
                      '(target not binary or single-class) - measurement is broken, '
                      'not necessarily a random model'),
            'mean_auc': None,
            'per_channel_auc': per_channel_auc,
            'per_channel_opt_acc': per_channel_opt_acc,
            'threshold': threshold,
        }

    mean_auc = float(np.mean(valid))
    if mean_auc < EXPL_ACC_FAIL:
        verdict = FAIL
    elif mean_auc < EXPL_ACC_WARN:
        verdict = WARN
    else:
        verdict = PASS

    result = {
        'verdict': verdict,
        'computed': True,
        'mean_auc': mean_auc,
        'per_channel_auc': per_channel_auc,
        'per_channel_opt_acc': per_channel_opt_acc,
        'threshold': threshold,
    }
    # partial coverage: some channels measurable, some not
    if uncomputable:
        result['uncomputable_channels'] = uncomputable
        result['note'] = (f'channels {uncomputable} could not be measured (non-binary or '
                          f'single-class proxy target); mean_auc is over the rest')
    return result


def _fidelity_sign_and_magnitude(model, raw: dict) -> t.Tuple[dict, dict, np.ndarray]:
    """
    Compute the fidelity sign-consistency and magnitude axes together (both derive
    from the leave-one-out deviations). Mirrors the trusted logic in
    ``MeganTrainingMetricsCallback._run_validation``.
    """
    # deviations: (B, O, K)
    deviations = raw['leave_one_out']
    values_true = raw['values_true']
    num_channels = model.num_channels

    # mean signed deviation per channel (averaged over graphs and targets)
    mean_dev = np.mean(deviations, axis=(0, 1))  # (K,)

    sign_consistency = np.zeros(num_channels)
    for k in range(num_channels):
        dev_k = deviations[:, :, k].mean(axis=1)  # (B,) mean over targets
        if model.prediction_mode == 'regression':
            # ch0 == "negative" (should push output down), ch1 == "positive"
            if k == 0:
                sign_consistency[k] = float(np.mean(dev_k < 0))
            else:
                sign_consistency[k] = float(np.mean(dev_k > 0))
        else:
            # each channel should positively contribute to its own class
            sign_consistency[k] = float(np.mean(dev_k > 0))

    worst_sign = float(np.min(sign_consistency))
    if worst_sign < SIGN_CONSISTENCY_FAIL:
        sign_verdict = FAIL
    elif worst_sign < SIGN_CONSISTENCY_WARN:
        sign_verdict = WARN
    else:
        sign_verdict = PASS

    sign_axis = {
        'verdict': sign_verdict,
        'per_channel_sign_consistency': [round(float(v), 4) for v in sign_consistency],
        'per_channel_mean_deviation': [round(float(v), 6) for v in mean_dev],
        'threshold': {'fail_below': SIGN_CONSISTENCY_FAIL, 'warn_below': SIGN_CONSISTENCY_WARN},
    }

    # -- magnitude --
    # target scale to make the magnitude interpretable/relative
    if model.prediction_mode == 'regression':
        target_scale = float(np.std(values_true)) or 1.0
    else:
        target_scale = 1.0

    mean_abs_dev = np.mean(np.abs(deviations), axis=(0, 1))  # (K,)
    relative = mean_abs_dev / target_scale
    mag_verdict = WARN if float(np.max(relative)) < FIDELITY_MAGNITUDE_WARN else PASS

    magnitude_axis = {
        'verdict': mag_verdict,
        'per_channel_mean_abs_deviation': [round(float(v), 6) for v in mean_abs_dev],
        'per_channel_relative_to_target_scale': [round(float(v), 4) for v in relative],
        'target_scale': round(target_scale, 6),
        'threshold': {'warn_below': FIDELITY_MAGNITUDE_WARN},
    }

    return sign_axis, magnitude_axis, deviations


def _axis_coverage(model, raw: dict) -> dict:
    """Soft / visual axis: how much of each graph each channel highlights."""
    results = raw['results']
    num_channels = model.num_channels

    per_channel = []
    verdict = PASS
    for k in range(num_channels):
        active_fracs: List[float] = []
        mean_imps: List[float] = []
        n_empty = 0
        n_saturated = 0
        for r in results:
            ni = np.asarray(r['node_importance'])[:, k]  # (V,)
            mean_imps.append(float(np.mean(ni)))
            mx = float(np.max(ni)) if ni.size else 0.0
            if mx < COVERAGE_EMPTY_EPS:
                n_empty += 1
                active_fracs.append(0.0)
                continue
            frac = float(np.mean((ni / mx) > COVERAGE_ACTIVE_THRESHOLD))
            active_fracs.append(frac)
            if frac > COVERAGE_SATURATED_FRAC_WARN:
                n_saturated += 1

        n = max(len(results), 1)
        entry = {
            'mean_importance': round(float(np.mean(mean_imps)) if mean_imps else 0.0, 4),
            'frac_nodes_active': round(float(np.mean(active_fracs)) if active_fracs else 0.0, 4),
            'frac_graphs_empty': round(n_empty / n, 4),
            'frac_graphs_saturated': round(n_saturated / n, 4),
        }
        per_channel.append(entry)

        # soft WARN at extremes only
        if entry['frac_graphs_empty'] > COVERAGE_EMPTY_FRAC_WARN:
            verdict = _worst(verdict, WARN)
        if entry['frac_nodes_active'] > COVERAGE_SATURATED_FRAC_WARN:
            verdict = _worst(verdict, WARN)

    return {
        'verdict': verdict,
        'per_channel': per_channel,
        'threshold': {
            'active_node_norm_above': COVERAGE_ACTIVE_THRESHOLD,
            'empty_frac_warn_above': COVERAGE_EMPTY_FRAC_WARN,
            'saturated_frac_warn_above': COVERAGE_SATURATED_FRAC_WARN,
        },
        'note': 'coverage is judged visually via examples.png; verdict here only flags extremes',
    }


def _axis_divergence(model, raw: dict, loss_history: Optional[List[float]]) -> dict:
    """Detect a diverged run: non-finite outputs or an exploded explanation loss."""
    values_pred = raw['values_pred']
    nan_in_output = not bool(np.all(np.isfinite(values_pred)))

    expl_loss_exploded = False
    final_expl_loss = None
    if loss_history:
        finite = [v for v in loss_history if np.isfinite(v)]
        final_expl_loss = float(loss_history[-1]) if loss_history else None
        # exploded == last value non-finite, or last >> best (10x the running min)
        if not np.isfinite(loss_history[-1]):
            expl_loss_exploded = True
        elif finite:
            best = min(finite)
            if best > 0 and loss_history[-1] > 10.0 * best:
                expl_loss_exploded = True

    verdict = FAIL if (nan_in_output or expl_loss_exploded) else PASS
    return {
        'verdict': verdict,
        'nan_in_output': nan_in_output,
        'expl_loss_exploded': expl_loss_exploded,
        'final_expl_loss': final_expl_loss,
    }


# == PUBLIC API ==

def compute_diagnostics(model,
                        graphs: List[dict],
                        loss_history: Optional[List[float]] = None,
                        raw: Optional[dict] = None,
                        ) -> dict:
    """
    Compute the full diagnostic report for a trained MEGAN ``model`` on ``graphs``.

    :param model: A trained ``Megan`` model instance (eval mode recommended).
    :param graphs: List of GraphDict elements to evaluate on (typically the test set).
    :param loss_history: Optional list of per-epoch explanation-loss values, used
        for divergence detection. When absent, only NaN-in-output is checked.
    :param raw: Optional pre-computed raw arrays (from ``_compute_raw``) to avoid
        recomputing the leave-one-out deviations.

    :returns: A JSON-serializable report dict with ``overall``, ``meta``,
        ``divergence`` and per-axis blocks under ``axes``. Each axis block carries
        the actual scalar values alongside its PASS/WARN/FAIL verdict.
    """
    if raw is None:
        raw = _compute_raw(model, graphs)

    prediction = _axis_prediction(model, raw)
    explanation_accuracy = _axis_explanation_accuracy(model, raw)
    fidelity_sign, fidelity_magnitude, _ = _fidelity_sign_and_magnitude(model, raw)
    coverage = _axis_coverage(model, raw)
    divergence = _axis_divergence(model, raw, loss_history)

    axes = {
        'prediction': prediction,
        'explanation_accuracy': explanation_accuracy,
        'fidelity_sign': fidelity_sign,
        'fidelity_magnitude': fidelity_magnitude,
        'coverage': coverage,
    }

    # -- overall verdict --
    # A run is FAILED if it diverged, if the explanation proxy task is ~random,
    # or if the fidelity sign is inverted. Those are the hard gates. Otherwise the
    # overall verdict is the worst of the remaining axes.
    hard_gates = {
        'divergence': divergence['verdict'],
        'explanation_accuracy': explanation_accuracy['verdict'],
        'fidelity_sign': fidelity_sign['verdict'],
    }
    overall_verdict = _worst(*[a['verdict'] for a in axes.values()], divergence['verdict'])

    reasons: List[str] = []
    if divergence['verdict'] == FAIL:
        why = 'NaN in output' if divergence['nan_in_output'] else 'explanation loss exploded'
        reasons.append(f'divergence FAIL: {why}')
    for name, axis in axes.items():
        if axis['verdict'] != PASS:
            reasons.append(f'{name} {axis["verdict"]}')

    channel_names = None
    reg_ref = None
    if model.prediction_mode == 'regression':
        try:
            reg_ref = float(np.asarray(model.regression_reference.detach().cpu()).flatten()[0])
        except Exception:
            reg_ref = None

    return {
        'overall': {
            'verdict': overall_verdict,
            'reasons': reasons,
            'hard_gates': hard_gates,
        },
        'meta': {
            'prediction_mode': model.prediction_mode,
            'num_channels': int(model.num_channels),
            'n_eval_graphs': len(graphs),
            'regression_reference': reg_ref,
        },
        'divergence': divergence,
        'axes': axes,
    }


def generate_diagnostic_report(model,
                               graphs: List[dict],
                               output_dir: str,
                               example_graphs: Optional[List[dict]] = None,
                               example_image_paths: Optional[List[str]] = None,
                               channel_infos: Optional[dict] = None,
                               num_targets: Optional[int] = None,
                               loss_history: Optional[List[float]] = None,
                               num_examples: int = NUM_EXAMPLES,
                               report_filename: str = 'report.json',
                               image_prefix: str = '',
                               ) -> dict:
    """
    Compute the diagnostics and write the full agent-facing artifact set into
    ``output_dir``: the JSON report plus ``fidelity.png``, ``examples.png`` and
    ``scorecard.png``.

    :param model: Trained ``Megan`` model.
    :param graphs: Graphs to evaluate on (the validation set for the tuning gate,
        or the test set for the final unbiased report).
    :param output_dir: Directory to write the artifacts into.
    :param report_filename: Name of the JSON report file (e.g. ``report.json`` for the
        validation gate, ``report_test.json`` for the final test report).
    :param image_prefix: Prefix for the image filenames (e.g. ``''`` -> ``scorecard.png``,
        ``test_`` -> ``test_scorecard.png``) so val and test artifacts don't collide.
    :param example_graphs: Graphs (with ``node_positions``) to draw explanations
        for in examples.png. If None, a random subset of ``graphs`` is used - but
        the image is only produced when valid image paths are also available.
    :param example_image_paths: Background image paths matching ``example_graphs``.
    :param channel_infos: Optional ``{k: {'name': .., 'color': ..}}`` mapping.
    :param num_targets: Number of model output targets (for the fidelity figure).
        Defaults to ``model.out_dim``.
    :param loss_history: Per-epoch explanation-loss values for divergence detection.
    :param num_examples: Number of example graphs to show (default 8).

    :returns: The report dict (also written to ``report.json``).
    """
    os.makedirs(output_dir, exist_ok=True)
    channel_infos = channel_infos or _default_channel_infos(model.num_channels)
    num_targets = num_targets or model.out_dim

    raw = _compute_raw(model, graphs)
    report = compute_diagnostics(model, graphs, loss_history=loss_history, raw=raw)

    # -- JSON report --
    with open(os.path.join(output_dir, report_filename), 'w') as f:
        json.dump(report, f, indent=2)

    # -- fidelity.png --
    try:
        fig = plot_leave_one_out_analysis(
            raw['leave_one_out'],
            num_channels=model.num_channels,
            num_targets=num_targets,
        )
        fig.savefig(os.path.join(output_dir, f'{image_prefix}fidelity.png'), bbox_inches='tight', dpi=120)
        plt.close(fig)
    except Exception:
        pass

    # -- examples.png --
    if example_graphs is None:
        example_graphs = graphs
        example_image_paths = None
    _render_examples(
        model=model,
        example_graphs=example_graphs,
        example_image_paths=example_image_paths,
        channel_infos=channel_infos,
        output_path=os.path.join(output_dir, f'{image_prefix}examples.png'),
        num_examples=num_examples,
    )

    # -- scorecard.png --
    _render_scorecard(
        report=report,
        channel_infos=channel_infos,
        output_path=os.path.join(output_dir, f'{image_prefix}scorecard.png'),
    )

    return report


# == IMAGE RENDERING ==

def _default_channel_infos(num_channels: int) -> dict:
    if num_channels == 2:
        return {0: {'name': 'negative', 'color': 'skyblue'},
                1: {'name': 'positive', 'color': 'coral'}}
    palette = plt.get_cmap('tab10')
    return {k: {'name': f'ch{k}', 'color': palette(k % 10)} for k in range(num_channels)}


def _render_examples(model,
                     example_graphs: List[dict],
                     example_image_paths: Optional[List[str]],
                     channel_infos: dict,
                     output_path: str,
                     num_examples: int,
                     ) -> None:
    """Grid of random example explanations: rows = channels, cols = examples."""
    n_total = len(example_graphs)
    if n_total == 0:
        return

    # require background images + node positions to draw explanations
    if example_image_paths is None or not all(p is not None for p in example_image_paths):
        return
    if not all('node_positions' in g for g in example_graphs):
        return

    n = min(num_examples, n_total)
    sel = np.random.choice(n_total, size=n, replace=False)

    sel_graphs = [example_graphs[i] for i in sel]
    sel_paths = [example_image_paths[i] for i in sel]
    infos = model.forward_graphs(sel_graphs)

    num_channels = model.num_channels
    fig, rows = plt.subplots(
        ncols=n, nrows=num_channels,
        figsize=(4 * n, 4 * num_channels),
        squeeze=False,
    )
    fig.suptitle('Random example explanations (rows = channels)', fontsize=12)

    for c, (graph, image_path, info) in enumerate(zip(sel_graphs, sel_paths, infos)):
        out_pred = np.asarray(info['graph_output']).flatten()
        out_true = np.asarray(graph['graph_labels']).flatten()
        for k in range(num_channels):
            ax = rows[k][c]
            ni = np.asarray(info['node_importance'])[:, k]
            ei = np.asarray(info['edge_importance'])[:, k]
            ax.set_title(
                f'ch{k} {channel_infos.get(k, {}).get("name", "")}\n'
                f'true {np.round(out_true, 2)} pred {np.round(out_pred, 2)}\n'
                f'imp mean {np.mean(ni):.2f} max {np.max(ni):.2f}',
                fontsize=7,
            )
            try:
                draw_image(ax=ax, image_path=image_path, remove_ticks=True)
                color = channel_infos.get(k, {}).get('color', 'coral')
                plot_node_importances_background(
                    ax=ax, g=graph, node_positions=graph['node_positions'],
                    node_importances=ni, color=color,
                )
                plot_edge_importances_background(
                    ax=ax, g=graph, node_positions=graph['node_positions'],
                    edge_importances=ei, color=color,
                )
            except Exception:
                ax.text(0.5, 0.5, 'render error', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches='tight', dpi=120)
    plt.close(fig)


_VERDICT_COLOR = {PASS: '#4CAF50', WARN: '#FF9800', FAIL: '#F44336'}


def _render_scorecard(report: dict,
                      channel_infos: dict,
                      output_path: str,
                      ) -> None:
    """One-glance panel: explanation-accuracy vs coverage per channel + verdicts."""
    axes_report = report['axes']
    num_channels = report['meta']['num_channels']
    ks = list(range(num_channels))
    names = [channel_infos.get(k, {}).get('name', f'ch{k}') for k in ks]

    fig, (ax_bar, ax_txt) = plt.subplots(
        ncols=2, nrows=1, figsize=(14, 6),
        gridspec_kw={'width_ratios': [1.4, 1.0]},
    )

    # -- left: accuracy (AUC) vs coverage (frac active) per channel --
    # per_channel_auc may be None (axis uncomputable) or contain None (per-channel);
    # substitute 0.0 for plotting so a broken/uncomputed metric reads as an empty bar.
    raw_auc = axes_report['explanation_accuracy'].get('per_channel_auc') or []
    auc = [(raw_auc[k] if k < len(raw_auc) and raw_auc[k] is not None else 0.0)
           for k in range(num_channels)]
    cov = [c['frac_nodes_active'] for c in axes_report['coverage']['per_channel']]
    x = np.arange(num_channels)
    w = 0.38
    ax_bar.bar(x - w / 2, auc, width=w, label='explanation AUC', color='#4CAF50')
    ax_bar.bar(x + w / 2, cov, width=w, label='coverage (frac nodes active)', color='#2196F3')
    ax_bar.axhline(EXPL_ACC_WARN, color='#4CAF50', ls='--', alpha=0.5, lw=1)
    ax_bar.axhline(EXPL_ACC_FAIL, color='#F44336', ls='--', alpha=0.5, lw=1)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title('Explanation accuracy vs coverage (antagonists)', fontsize=10)
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, axis='y', alpha=0.3)

    # -- right: verdict summary text --
    ax_txt.axis('off')
    overall = report['overall']['verdict']
    y = 0.98
    ax_txt.text(0.0, y, f'OVERALL: {overall}', fontsize=16, fontweight='bold',
                color=_VERDICT_COLOR[overall], transform=ax_txt.transAxes)
    y -= 0.10
    order = ['divergence', 'prediction', 'explanation_accuracy',
             'fidelity_sign', 'fidelity_magnitude', 'coverage']
    for name in order:
        block = report['divergence'] if name == 'divergence' else axes_report[name]
        v = block['verdict']
        ax_txt.text(0.0, y, f'{name}', fontsize=10, transform=ax_txt.transAxes)
        ax_txt.text(0.62, y, v, fontsize=10, fontweight='bold',
                    color=_VERDICT_COLOR[v], transform=ax_txt.transAxes)
        y -= 0.075
    if report['overall']['reasons']:
        y -= 0.03
        ax_txt.text(0.0, y, 'reasons:', fontsize=9, fontstyle='italic',
                    transform=ax_txt.transAxes)
        y -= 0.06
        for r in report['overall']['reasons']:
            ax_txt.text(0.02, y, f'- {r}', fontsize=8, transform=ax_txt.transAxes)
            y -= 0.05

    fig.suptitle('MEGAN post-training scorecard', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, bbox_inches='tight', dpi=120)
    plt.close(fig)
