"""Combined scoring: frozen fit-window weights + rank-based AUC.

Weight formula (stated, frozen, no optimiser, no fitting on held-out):

    w_s = ln(lift_fit_s)   if lift_fit_s > 1 and s is not leakage-flagged
    w_s = 0                otherwise

Weights are derived from the FIT window only. A leakage-flagged
signal is excluded from the combined score entirely (weight 0 and
listed in ``excluded``), however good it looks in-sample. A
decay-flagged signal is *not* excluded — decay is reported, and the
held-out AUC shows what the decay costs.

AUC is the rank-based Mann-Whitney statistic (average ranks for
ties), stdlib only. No PR curves, no calibration plots.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from generator import SIGNAL_NAMES, WINDOWS
from signals import cohort_and_label


def compute_weights(fit_metrics: Dict[str, Dict],
                    excluded: Sequence[str]) -> Dict[str, float]:
    """fit_metrics: {signal: metrics-dict with a 'lift' key}."""
    weights = {}
    for signal in SIGNAL_NAMES:
        lift = fit_metrics[signal]["lift"]
        if signal in excluded or lift <= 1.0:
            weights[signal] = 0.0
        else:
            weights[signal] = math.log(lift)
    return weights


def combined_score(firing: Dict[str, bool],
                   weights: Dict[str, float]) -> float:
    return sum(w for s, w in weights.items() if firing.get(s))


def auc(scores_labels: List[Tuple[float, bool]]) -> float:
    """Mann-Whitney AUC with average ranks for tied scores."""
    if not scores_labels:
        raise ValueError("auc needs at least one scored account")
    n_pos = sum(1 for _s, label in scores_labels if label)
    n_neg = len(scores_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auc needs both positive and negative labels")
    ordered = sorted(scores_labels, key=lambda p: p[0])
    ranks: Dict[int, float] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][0] == ordered[i][0]:
            j += 1
        average_rank = (i + 1 + j + 1) / 2.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[k] = average_rank
        i = j + 1
    rank_sum_pos = sum(ranks[k] for k, (_s, label) in enumerate(ordered)
                       if label)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def score_window(dataset, signals_comp, weights, window):
    """Per-account combined scores for one window's cohort."""
    ids, labels = cohort_and_label(dataset, window)
    rows = [{"account_id": account_id,
             "score": round(combined_score(
                 signals_comp[account_id][window], weights), 4),
             "converted_in_window": labels[account_id]}
            for account_id in ids]
    return rows


def window_auc(dataset, signals_comp, weights, window) -> float:
    ids, labels = cohort_and_label(dataset, window)
    pairs = [(combined_score(signals_comp[a][window], weights), labels[a])
             for a in ids]
    return round(auc(pairs), 4)


def single_signal_aucs(dataset, signals_comp, window,
                       signals=SIGNAL_NAMES) -> Dict[str, float]:
    ids, labels = cohort_and_label(dataset, window)
    out = {}
    for signal in signals:
        pairs = [(1.0 if signals_comp[a][window][signal] else 0.0,
                  labels[a]) for a in ids]
        out[signal] = round(auc(pairs), 4)
    return out


def score_dataset(dataset, signals_comp, weights,
                  excluded: Sequence[str] = ()) -> Dict:
    """Scores + AUCs for both windows, plus the single-signal baseline.

    The baseline is the best held-out AUC of any single non-excluded
    signal used alone — the bar the combined score has to beat.
    """
    per_window = {}
    for window in WINDOWS:
        per_window[window] = {
            "auc": window_auc(dataset, signals_comp, weights, window),
            "single_signal_aucs": single_signal_aucs(
                dataset, signals_comp, window),
            "scores": score_window(dataset, signals_comp, weights, window),
        }
    candidates = [s for s in SIGNAL_NAMES if s not in excluded]
    baseline_signal = max(
        candidates,
        key=lambda s: per_window["heldout"]["single_signal_aucs"][s])
    per_window["baseline"] = {
        "signal": baseline_signal,
        "heldout_auc": per_window["heldout"]["single_signal_aucs"]
        [baseline_signal],
        "definition": "best single non-leakage signal, held-out AUC",
    }
    return per_window
