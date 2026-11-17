"""End-to-end demo: generate -> signals -> leakage/decay checks ->
score -> report -> check exit codes -> serve.

SYNTHETIC DATA ONLY (fictional vendor and accounts). Runs fully
offline; the serve round-trip is loopback-only on an ephemeral port.
Every stage asserts.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import EXIT_ERROR, EXIT_FAIL, EXIT_OK, main as cli_main, \
    make_server, score_one  # noqa: E402
from generator import generate_dataset  # noqa: E402
from report import build_report  # noqa: E402
from signals import compute_signals  # noqa: E402


def _request(port, path, payload=None, method="GET"):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def main() -> None:
    print("== 1. generate: 600 fictional accounts, 26 weeks ==")
    ds = generate_dataset()
    assert ds == generate_dataset(), "generator must be deterministic"
    truth = ds["ground_truth"]
    assert truth["counts"] == {"accounts": 600, "fit_converters": 90,
                               "heldout_converters": 90,
                               "never_converters": 420}
    print(f"   {len(ds['accounts'])} accounts, {len(ds['events'])} "
          f"timestamped events, {len(ds['weekly_activity'])} weekly "
          f"activity rows; 90 fit + 90 held-out converters")

    print("\n== 2. signals recomputed from activity == planted ==")
    comp = compute_signals(ds)
    planted = truth["planted_firing"]
    mismatches = [(a, w, s) for a in planted for w in planted[a]
                  for s in planted[a][w]
                  if planted[a][w][s] != comp[a][w][s]]
    assert not mismatches, mismatches[:3]
    print("   all 600 x 2 windows x 5 signals match the planted firing")

    print("\n== 3. the two ways signal scores lie ==")
    report = build_report(ds)
    assert report["flags"] == {"leakage": ["contract_doc_download"],
                               "decay": ["webinar_attendance"]}, report["flags"]
    leak = report["signals"]["contract_doc_download"]["leakage"]
    decay = report["signals"]["webinar_attendance"]["decay"]
    print(f"   LEAKAGE contract_doc_download: {leak['post_conversion_timestamps']}"
          f" post-conversion timestamps — looks like lift "
          f"{report['signals']['contract_doc_download']['fit']['lift']} "
          f"in-sample, and every one of those events happened AFTER the "
          f"deal closed. Excluded from the score.")
    print(f"   DECAY webinar_attendance: fit lift {decay['fit_lift']} -> "
          f"held-out lift {decay['heldout_lift']} (ratio "
          f"{decay['heldout_to_fit_ratio']}, threshold 0.4)")
    for s in ("product_usage_spike", "pricing_page_visits",
              "champion_engagement"):
        sig = report["signals"][s]
        assert not sig["leakage"]["flagged_leakage"]
        assert not sig["decay"]["flagged_decay"]
    print("   the 3 honest signals are flagged by neither check")

    print("\n== 4. combined score: fit-only weights, leakage excluded ==")
    print(f"   weights (ln of fit lift): {report['weights']}")
    assert report["weights"]["contract_doc_download"] == 0.0
    c = report["combined_score"]
    print(f"   AUC fit={c['fit_auc']} held-out={c['heldout_auc']} vs "
          f"best single signal {c['baseline']['signal']} "
          f"{c['baseline']['heldout_auc']} held-out")
    assert c["heldout_auc"] > c["baseline"]["heldout_auc"]

    print("\n== 5. CLI check exit codes: 1 seeded / 0 clean / 2 bad ==")
    tmp = tempfile.mkdtemp(prefix="gtm-signal-demo-")
    seeded_path = os.path.join(tmp, "seeded.json")
    clean_path = os.path.join(tmp, "clean.json")
    assert cli_main(["generate", "--out", seeded_path]) == EXIT_OK
    assert cli_main(["generate", "--out", clean_path,
                     "--variant", "clean"]) == EXIT_OK
    code_seeded = cli_main(["check", "--input", seeded_path])
    code_clean = cli_main(["check", "--input", clean_path])
    code_bad = cli_main(["check", "--input", "/tmp/no-such-ds-xyz.json"])
    print(f"   seeded={code_seeded} clean={code_clean} missing={code_bad}")
    assert (code_seeded, code_clean, code_bad) == (EXIT_FAIL, EXIT_OK,
                                                   EXIT_ERROR)

    print("\n== 6. live localhost serve round-trip ==")
    server = make_server(0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, health = _request(port, "/health")
        assert status == 200 and health == {"status": "ok"}
        sample = {"product_usage_spike": True, "pricing_page_visits": True,
                  "champion_engagement": False, "webinar_attendance": False,
                  "contract_doc_download": True}
        status, served = _request(port, "/score",
                                  payload={"signals": sample}, method="POST")
        assert status == 200 and served == score_one(sample), served
        assert served["excluded_signals"] == ["contract_doc_download"]
        assert served["contributions"]["contract_doc_download"] == 0.0
        print(f"   POST /score -> {served['score']} (the leakage signal "
              f"fired and still contributed 0.0)")
    finally:
        server.shutdown()
        server.server_close()

    print("\nDemo OK: generate, signals, leakage, decay, score, report, "
          "check and serve verified.")


if __name__ == "__main__":
    main()
