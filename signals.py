"""Signal computation, per-window metrics, leakage and decay checks.

Signals are computed ONLY from activity (weekly sessions + timestamped
events). Nothing here reads the generator's planted firing flags —
the evals prove the recomputation matches them.

Definitions (thresholds are the whole signal; no hidden logic):

* product_usage_spike   — any week in the window with sessions >= 6
* pricing_page_visits   — >= 3 pricing_page_view events in the window
* champion_engagement   — >= 2 champion_engagement events in window
* webinar_attendance    — >= 1 webinar_attendance event in the window
* contract_doc_download — >= 1 contract_doc_download event in window

Per-window evaluation cohort: accounts not yet converted at the
window start. Label: the account converts inside the window.
Lift = precision / base rate, where base rate is the cohort's
conversion rate in that window.

Leakage check: every timestamp that feeds a signal must precede the
account's conversion timestamp. A signal with any post-conversion
timestamp is flagged LEAKAGE with the offending count.

Decay check: a signal is flagged DECAY when its held-out lift falls
below DECAY_RATIO (0.40) of its fit lift (and fit lift was >= 1.5,
so noise on a useless signal is not called decay).
"""

from __future__ import annotations

from typing import Dict, List

from generator import SIGNAL_NAMES, WINDOWS

DECAY_RATIO = 0.40
DECAY_MIN_FIT_LIFT = 1.5
SPIKE_SESSION_THRESHOLD = 6
PRICING_THRESHOLD = 3
CHAMPION_THRESHOLD = 2

_EVENT_TYPE_FOR = {
    "pricing_page_visits": "pricing_page_view",
    "champion_engagement": "champion_engagement",
    "webinar_attendance": "webinar_attendance",
    "contract_doc_download": "contract_doc_download",
}
THRESHOLDS = {
    "product_usage_spike": "max weekly sessions >= 6",
    "pricing_page_visits": "pricing_page_view events >= 3",
    "champion_engagement": "champion_engagement events >= 2",
    "webinar_attendance": "webinar_attendance events >= 1",
    "contract_doc_download": "contract_doc_download events >= 1",
}


def _window_of_day(day: int):
    for name, (start, end) in WINDOWS.items():
        if start <= day <= end:
            return name
    return None


def compute_signals(dataset: Dict) -> Dict[str, Dict[str, Dict[str, bool]]]:
    """{account_id: {window: {signal: fired}}} from activity only."""
    out = {a["account_id"]: {w: {s: False for s in SIGNAL_NAMES}
                             for w in WINDOWS}
           for a in dataset["accounts"]}
    max_sessions: Dict = {}
    for row in dataset["weekly_activity"]:
        window = "fit" if row["week"] <= 12 else "heldout"
        key = (row["account_id"], window)
        max_sessions[key] = max(max_sessions.get(key, 0), row["sessions"])
    for (account_id, window), peak in max_sessions.items():
        out[account_id][window]["product_usage_spike"] = (
            peak >= SPIKE_SESSION_THRESHOLD)
    counts: Dict = {}
    for event in dataset["events"]:
        window = _window_of_day(event["day"])
        if window is None:
            continue
        key = (event["account_id"], window, event["event_type"])
        counts[key] = counts.get(key, 0) + 1
    for signal, event_type in _EVENT_TYPE_FOR.items():
        threshold = {"pricing_page_visits": PRICING_THRESHOLD,
                     "champion_engagement": CHAMPION_THRESHOLD}.get(
                         signal, 1)
        for a in dataset["accounts"]:
            account_id = a["account_id"]
            for window in WINDOWS:
                if counts.get((account_id, window, event_type), 0) >= threshold:
                    out[account_id][window][signal] = True
    return out


def signal_timestamps(dataset: Dict, signal: str) -> Dict[str, List[int]]:
    """All timestamps (days) feeding ``signal``, per account."""
    per_account: Dict[str, List[int]] = {
        a["account_id"]: [] for a in dataset["accounts"]}
    if signal == "product_usage_spike":
        for row in dataset["weekly_activity"]:
            if row["sessions"] >= SPIKE_SESSION_THRESHOLD:
                per_account[row["account_id"]].append(row["week"] * 7)
        return per_account
    event_type = _EVENT_TYPE_FOR[signal]
    for event in dataset["events"]:
        if event["event_type"] == event_type:
            per_account[event["account_id"]].append(event["day"])
    return per_account


def leakage_check(dataset: Dict) -> Dict[str, Dict]:
    """Per signal: post-conversion timestamps and the LEAKAGE flag."""
    conversions = {a["account_id"]: a["conversion_day"]
                   for a in dataset["accounts"]}
    result = {}
    for signal in SIGNAL_NAMES:
        stamps = signal_timestamps(dataset, signal)
        total = sum(len(v) for v in stamps.values())
        offending = sum(
            1 for account_id, days in stamps.items()
            if conversions[account_id] is not None
            for d in days if d > conversions[account_id])
        result[signal] = {
            "signal_timestamps": total,
            "post_conversion_timestamps": offending,
            "flagged_leakage": offending > 0,
        }
    return result


def cohort_and_label(dataset: Dict, window: str):
    """(account_ids in cohort, {account_id: converted_in_window})."""
    start, end = WINDOWS[window]
    ids, labels = [], {}
    for a in dataset["accounts"]:
        conv = a["conversion_day"]
        if conv is not None and conv < start:
            continue  # already converted before this window opened
        ids.append(a["account_id"])
        labels[a["account_id"]] = conv is not None and start <= conv <= end
    return ids, labels


def _round4(x: float) -> float:
    return round(x, 4)


def signal_metrics(dataset: Dict, signals_comp: Dict,
                   window: str, signal: str) -> Dict:
    ids, labels = cohort_and_label(dataset, window)
    tp = fp = fn = tn = 0
    for account_id in ids:
        fired = signals_comp[account_id][window][signal]
        label = labels[account_id]
        if fired and label:
            tp += 1
        elif fired:
            fp += 1
        elif label:
            fn += 1
        else:
            tn += 1
    total = len(ids)
    positives = tp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / positives if positives else 0.0
    base_rate = positives / total if total else 0.0
    lift = precision / base_rate if base_rate else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "cohort": total,
            "positives": positives,
            "precision": _round4(precision), "recall": _round4(recall),
            "base_rate": _round4(base_rate), "lift": _round4(lift)}


def all_metrics(dataset: Dict, signals_comp: Dict) -> Dict:
    return {signal: {window: signal_metrics(dataset, signals_comp,
                                            window, signal)
                     for window in WINDOWS}
            for signal in SIGNAL_NAMES}


def decay_check(metrics: Dict) -> Dict[str, Dict]:
    """Per signal: fit vs held-out lift, ratio, and the DECAY flag."""
    result = {}
    for signal in SIGNAL_NAMES:
        fit_lift = metrics[signal]["fit"]["lift"]
        held_lift = metrics[signal]["heldout"]["lift"]
        ratio = _round4(held_lift / fit_lift) if fit_lift else 0.0
        result[signal] = {
            "fit_lift": fit_lift, "heldout_lift": held_lift,
            "heldout_to_fit_ratio": ratio,
            "flagged_decay": (fit_lift >= DECAY_MIN_FIT_LIFT
                              and ratio < DECAY_RATIO),
        }
    return result


def flags_from(leakage: Dict, decay: Dict) -> Dict[str, List[str]]:
    return {
        "leakage": [s for s in SIGNAL_NAMES if leakage[s]["flagged_leakage"]],
        "decay": [s for s in SIGNAL_NAMES if decay[s]["flagged_decay"]],
    }
