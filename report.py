"""Combined report: metrics, leakage/decay flags, weights, AUCs."""

from __future__ import annotations

from typing import Dict

from generator import SIGNAL_NAMES
from scoring import compute_weights, score_dataset
from signals import all_metrics, compute_signals, decay_check, flags_from, \
    leakage_check


def build_report(dataset: Dict) -> Dict:
    signals_comp = compute_signals(dataset)
    metrics = all_metrics(dataset, signals_comp)
    leakage = leakage_check(dataset)
    decay = decay_check(metrics)
    flags = flags_from(leakage, decay)
    fit_metrics = {s: metrics[s]["fit"] for s in SIGNAL_NAMES}
    weights = compute_weights(fit_metrics, excluded=flags["leakage"])
    scored = score_dataset(dataset, signals_comp, weights,
                           excluded=flags["leakage"])
    fit_auc = scored["fit"]["auc"]
    held_auc = scored["heldout"]["auc"]
    baseline = scored["baseline"]
    narrative = (
        f"On {len(dataset['accounts'])} synthetic accounts, the "
        f"leakage-free combined score reaches AUC {fit_auc} on the fit "
        f"window and {held_auc} held-out, versus {baseline['heldout_auc']} "
        f"for the best single signal ({baseline['signal']}). "
        f"Leakage flags: {flags['leakage'] or 'none'} — a leakage-flagged "
        f"signal is excluded from the score no matter how predictive it "
        f"looks in-sample. Decay flags: {flags['decay'] or 'none'} — its "
        f"fit-window lift did not survive the held-out window.")
    return {
        "vendor": dataset["meta"]["vendor"],
        "variant": dataset["meta"]["variant"],
        "seed": dataset["meta"]["seed"],
        "accounts": len(dataset["accounts"]),
        "events": len(dataset["events"]),
        "windows": dataset["meta"]["windows"],
        "signals": {
            s: {"fit": metrics[s]["fit"], "heldout": metrics[s]["heldout"],
                "leakage": leakage[s], "decay": decay[s]}
            for s in SIGNAL_NAMES},
        "flags": flags,
        "weight_formula": "w_s = ln(lift_fit_s) if lift_fit_s > 1 and "
                          "s not leakage-flagged, else 0",
        "weights": {s: round(w, 4) for s, w in weights.items()},
        "excluded_from_score": flags["leakage"],
        "combined_score": {
            "fit_auc": fit_auc,
            "heldout_auc": held_auc,
            "baseline": baseline,
            "fit_single_signal_aucs": scored["fit"]["single_signal_aucs"],
            "heldout_single_signal_aucs":
                scored["heldout"]["single_signal_aucs"],
        },
        "narrative": narrative,
    }
