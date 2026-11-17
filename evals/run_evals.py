"""Golden evals for gtm-signal-scoring-eval-kit.

SYNTHETIC DATA ONLY (fictional vendor and accounts). Every eval is
deterministic: run_evals.py writes eval_report.json and the file
must be byte-identical across runs. No timestamps, ports, or temp
paths enter the report — only names, booleans, pinned counts and
values.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
import tempfile
import threading
import urllib.request
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import EXIT_ERROR, EXIT_FAIL, EXIT_OK, main as cli_main, \
    make_server, score_one  # noqa: E402
from generator import SIGNAL_NAMES, dataset_to_json, generate_dataset  # noqa: E402
from report import build_report  # noqa: E402
from scoring import compute_weights  # noqa: E402
from signals import compute_signals  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = []

# Pinned headline values (seed 42, seeded variant).
PINNED_LIFTS = {  # signal: (fit lift, held-out lift)
    "product_usage_spike": (2.7273, 3.0909),
    "pricing_page_visits": (3.003, 2.9403),
    "champion_engagement": (2.8188, 2.2842),
    "webinar_attendance": (3.6915, 1.02),
    "contract_doc_download": (5.9028, 4.9026),
}
PINNED_WEIGHTS = {"product_usage_spike": 1.0033,
                  "pricing_page_visits": 1.0996,
                  "champion_engagement": 1.0363,
                  "webinar_attendance": 1.306,
                  "contract_doc_download": 0.0}
PINNED_FIT_AUC = 0.9476
PINNED_HELDOUT_AUC = 0.8734
PINNED_BASELINE = {"signal": "product_usage_spike", "heldout_auc": 0.7738}
PINNED_LEAKAGE_OFFENDING = 162
PINNED_DATASET_MD5 = "12b9ab57e866ad01ee7e2dce75ba4b6f"


def check(name, passed, detail=""):
    RESULTS.append({"name": name, "passed": bool(passed), "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
          + (f" — {detail}" if detail else ""))
    return passed


def _run_cli(argv):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return cli_main(argv)


def main():
    print("== golden evals: gtm-signal-scoring-eval-kit ==")
    ds = generate_dataset()
    report = build_report(ds)

    # E1: generator determinism (canonical bytes pinned by md5).
    text1 = dataset_to_json(generate_dataset())
    text2 = dataset_to_json(generate_dataset())
    md5 = hashlib.md5(text1.encode()).hexdigest()
    check("E1 generator determinism (seed=42, 600 accounts)",
          text1 == text2 and md5 == PINNED_DATASET_MD5
          and len(ds["accounts"]) == 600 and len(ds["events"]) == 3429,
          f"dataset_md5={md5}")

    # E2: dataset ground-truth counts pinned.
    counts = ds["ground_truth"]["counts"]
    check("E2 ground truth pinned (90 fit / 90 held-out / 420 never)",
          counts == {"accounts": 600, "fit_converters": 90,
                     "heldout_converters": 90, "never_converters": 420})

    # E3: leakage flagged exactly, with the pinned offending count,
    # and the signal is excluded from the combined score.
    leak = report["signals"]["contract_doc_download"]["leakage"]
    check("E3 leakage flagged exactly (contract_doc_download, excluded)",
          report["flags"]["leakage"] == ["contract_doc_download"]
          and leak["post_conversion_timestamps"] == PINNED_LEAKAGE_OFFENDING
          and leak["flagged_leakage"]
          and report["excluded_from_score"] == ["contract_doc_download"]
          and report["weights"]["contract_doc_download"] == 0.0,
          f"offending={leak['post_conversion_timestamps']}")

    # E4: decay flagged exactly, ratio pinned below the 0.40 threshold.
    decay = report["signals"]["webinar_attendance"]["decay"]
    check("E4 decay flagged exactly (webinar_attendance, ratio 0.2763)",
          report["flags"]["decay"] == ["webinar_attendance"]
          and decay["heldout_to_fit_ratio"] == 0.2763
          and decay["flagged_decay"],
          f"fit_lift={decay['fit_lift']} heldout_lift={decay['heldout_lift']}")

    # E5: the 3 honest signals are flagged by neither check.
    honest_clean = all(
        not report["signals"][s]["leakage"]["flagged_leakage"]
        and not report["signals"][s]["decay"]["flagged_decay"]
        and report["signals"][s]["leakage"]["post_conversion_timestamps"] == 0
        for s in ("product_usage_spike", "pricing_page_visits",
                  "champion_engagement"))
    check("E5 honest signals flagged by neither check", honest_clean)

    # E6: weights pinned AND equal to the stated formula recomputed
    # from the fit-window lifts (ln of lift, leakage excluded).
    fit_metrics = {s: report["signals"][s]["fit"] for s in SIGNAL_NAMES}
    recomputed = compute_weights(fit_metrics,
                                 excluded=report["flags"]["leakage"])
    formula_ok = all(
        abs(recomputed[s] - (0.0 if s == "contract_doc_download"
                             else math.log(fit_metrics[s]["lift"]))) < 1e-12
        for s in SIGNAL_NAMES)
    check("E6 fit-window weights pinned + match ln(fit lift) formula",
          report["weights"] == PINNED_WEIGHTS and formula_ok,
          f"weights={report['weights']}")

    # E7: AUCs pinned; held-out AUC beats the single-signal baseline
    # and sits in the sane band.
    c = report["combined_score"]
    check("E7 AUCs pinned; held-out beats single-signal baseline",
          c["fit_auc"] == PINNED_FIT_AUC
          and c["heldout_auc"] == PINNED_HELDOUT_AUC
          and c["baseline"]["signal"] == PINNED_BASELINE["signal"]
          and c["baseline"]["heldout_auc"] == PINNED_BASELINE["heldout_auc"]
          and c["heldout_auc"] > c["baseline"]["heldout_auc"]
          and 0.65 < c["heldout_auc"] < 0.95,
          f"fit={c['fit_auc']} heldout={c['heldout_auc']} "
          f"baseline={c['baseline']['heldout_auc']}")

    # E8: per-signal fit vs held-out lifts pinned for all 5 signals.
    lifts_ok = all(
        (report["signals"][s]["fit"]["lift"],
         report["signals"][s]["heldout"]["lift"]) == PINNED_LIFTS[s]
        for s in SIGNAL_NAMES)
    check("E8 per-signal lifts pinned (fit vs held-out, all 5)",
          lifts_ok,
          "webinar 3.6915->1.02; contract 5.9028->4.9026 (in-sample mirage)")

    # E9: CLI check exit codes — 1 on seeded (by design), 0 on the
    # clean variant, 2 on a missing file.
    tmp = tempfile.mkdtemp(prefix="gtm-signal-evals-")
    seeded_path = os.path.join(tmp, "seeded.json")
    clean_path = os.path.join(tmp, "clean.json")
    _run_cli(["generate", "--out", seeded_path])
    _run_cli(["generate", "--out", clean_path, "--variant", "clean"])
    codes = (_run_cli(["check", "--input", seeded_path]),
             _run_cli(["check", "--input", clean_path]),
             _run_cli(["check", "--input", "/tmp/no-such-ds-xyz.json"]))
    check("E9 CLI check exit codes 1 seeded / 0 clean / 2 missing",
          codes == (EXIT_FAIL, EXIT_OK, EXIT_ERROR), f"codes={codes}")

    # E10: serve round-trip — /score matches local scoring, the
    # leakage signal contributes 0 even when it fires, /health ok.
    server = make_server(0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                    timeout=10) as resp:
            health = json.loads(resp.read())

        def _post(payload):
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/score",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())

        sample = {s: False for s in SIGNAL_NAMES}
        sample["contract_doc_download"] = True
        sample["pricing_page_visits"] = True
        served = _post({"signals": sample})
        serve_ok = health == {"status": "ok"} and served == score_one(sample) \
            and served["contributions"]["contract_doc_download"] == 0.0
    finally:
        server.shutdown()
        server.server_close()
    check("E10 serve round-trip matches local score; leakage adds 0",
          serve_ok)

    # E11: signals recomputed from activity equal the planted firing
    # for every account, window and signal (600 x 2 x 5).
    comp = compute_signals(ds)
    planted = ds["ground_truth"]["planted_firing"]
    check("E11 recomputed signals == planted firing (600x2x5)",
          all(planted[a][w] == comp[a][w] for a in planted for w in planted[a]))

    # E12: report determinism — canonical JSON identical across builds.
    r1 = json.dumps(build_report(generate_dataset()), sort_keys=True)
    r2 = json.dumps(build_report(generate_dataset()), sort_keys=True)
    check("E12 report determinism (canonical JSON identical)",
          r1 == r2, f"report_md5={hashlib.md5(r1.encode()).hexdigest()}")

    passed = sum(1 for r in RESULTS if r["passed"])
    total_count = len(RESULTS)
    summary_doc = {"passed": passed, "total": total_count,
                   "all_passed": passed == total_count, "evals": RESULTS}
    with open(os.path.join(HERE, "eval_report.json"), "w",
              encoding="utf-8") as fh:
        fh.write(json.dumps(summary_doc, sort_keys=True) + "\n")
    print(f"\n{passed}/{total_count} evals passed -> evals/eval_report.json")
    return 0 if passed == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
