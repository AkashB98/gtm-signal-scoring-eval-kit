"""CLI: generate | signals | score | report | check | demo | serve.

SYNTHETIC DATA ONLY — every dataset handled here is fabricated
sample data (fictional vendor "Meridian Metrics", fictional accounts).

Exit codes:
  check : 0 = clean (no leakage, no decay), 1 = leakage and/or decay
          flagged — BY DESIGN the seeded dataset exits 1, because it
          plants both; a clean-variant dataset exits 0 —
          2 = usage / IO / validation error
  others: 0 = success, 2 = usage / IO / validation error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generator import DEFAULT_SEED, generate_dataset  # noqa: E402
from report import build_report  # noqa: E402
from scoring import combined_score, compute_weights  # noqa: E402
from signals import all_metrics, compute_signals, decay_check, flags_from, \
    leakage_check  # noqa: E402
from validation import validate_dataset, validate_score_request  # noqa: E402

EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2

_FROZEN = None  # cached (weights, excluded) from the seeded dataset


def canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True)


def load_dataset_file(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    validate_dataset(data)
    return data


def _load_or_error(path: Optional[str]):
    """Dataset from ``path``, or the seeded default when path is None."""
    if path is None:
        return generate_dataset()
    try:
        return load_dataset_file(path)
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
    except (ValueError, OSError) as exc:
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
    return None


def _write_out(args, payload, summary: str) -> int:
    text = canonical(payload) + "\n"
    if getattr(args, "out", None):
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as exc:
            print(f"error: cannot write {args.out}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(summary)
    else:
        print(text, end="")
    return EXIT_OK


def frozen_weights():
    """Weights + exclusions derived once from the seeded fit window."""
    global _FROZEN
    if _FROZEN is None:
        dataset = generate_dataset()
        comp = compute_signals(dataset)
        metrics = all_metrics(dataset, comp)
        flags = flags_from(leakage_check(dataset), decay_check(metrics))
        from generator import SIGNAL_NAMES
        fit_metrics = {s: metrics[s]["fit"] for s in SIGNAL_NAMES}
        weights = compute_weights(fit_metrics, excluded=flags["leakage"])
        _FROZEN = (weights, flags["leakage"])
    return _FROZEN


def score_one(signals: dict) -> dict:
    weights, excluded = frozen_weights()
    firing = validate_score_request({"signals": signals})
    contributions = {s: (round(weights[s], 4) if firing[s] else 0.0)
                     for s in weights}
    return {"score": round(combined_score(firing, weights), 4),
            "weights": {s: round(w, 4) for s, w in weights.items()},
            "contributions": contributions,
            "excluded_signals": excluded}


def cmd_generate(args) -> int:
    try:
        dataset = generate_dataset(seed=args.seed, variant=args.variant)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    try:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(dataset, sort_keys=True) + "\n")
        if args.truth:
            with open(args.truth, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(dataset["ground_truth"],
                                    sort_keys=True) + "\n")
    except OSError as exc:
        print(f"error: cannot write output: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(f"generate: {len(dataset['accounts'])} accounts, "
          f"{len(dataset['events'])} events, variant={args.variant} "
          f"seed={args.seed} -> {args.out}")
    return EXIT_OK


def cmd_signals(args) -> int:
    dataset = _load_or_error(args.input)
    if dataset is None:
        return EXIT_ERROR
    comp = compute_signals(dataset)
    metrics = all_metrics(dataset, comp)
    payload = {"signals": comp, "metrics": metrics}
    return _write_out(args, payload,
                      f"signals: {len(comp)} accounts -> {args.out}")


def cmd_score(args) -> int:
    dataset = _load_or_error(args.input)
    if dataset is None:
        return EXIT_ERROR
    from scoring import score_dataset
    from generator import SIGNAL_NAMES
    comp = compute_signals(dataset)
    metrics = all_metrics(dataset, comp)
    flags = flags_from(leakage_check(dataset), decay_check(metrics))
    fit_metrics = {s: metrics[s]["fit"] for s in SIGNAL_NAMES}
    weights = compute_weights(fit_metrics, excluded=flags["leakage"])
    scored = score_dataset(dataset, comp, weights,
                           excluded=flags["leakage"])
    payload = {"weights": {s: round(w, 4) for s, w in weights.items()},
               "excluded_from_score": flags["leakage"],
               "fit_auc": scored["fit"]["auc"],
               "heldout_auc": scored["heldout"]["auc"],
               "baseline": scored["baseline"],
               "fit_scores": scored["fit"]["scores"],
               "heldout_scores": scored["heldout"]["scores"]}
    return _write_out(args, payload,
                      f"score: fit_auc={payload['fit_auc']} "
                      f"heldout_auc={payload['heldout_auc']} -> {args.out}")


def cmd_report(args) -> int:
    dataset = _load_or_error(args.input)
    if dataset is None:
        return EXIT_ERROR
    report = build_report(dataset)
    return _write_out(args, report, f"report -> {args.out}")


def cmd_check(args) -> int:
    dataset = _load_or_error(args.input)
    if dataset is None:
        return EXIT_ERROR
    report = build_report(dataset)
    flags = report["flags"]
    print(f"accounts={report['accounts']} "
          f"fit_auc={report['combined_score']['fit_auc']} "
          f"heldout_auc={report['combined_score']['heldout_auc']} "
          f"leakage={flags['leakage']} decay={flags['decay']}")
    breaches = []
    for signal in flags["leakage"]:
        count = report["signals"][signal]["leakage"][
            "post_conversion_timestamps"]
        breaches.append(f"LEAKAGE {signal}: {count} post-conversion "
                        f"timestamps — excluded from the score")
    for signal in flags["decay"]:
        d = report["signals"][signal]["decay"]
        breaches.append(f"DECAY {signal}: held-out lift {d['heldout_lift']} "
                        f"is {d['heldout_to_fit_ratio']} of fit lift "
                        f"{d['fit_lift']} (threshold 0.4)")
    if breaches:
        for breach in breaches:
            print(f"check FAIL: {breach}")
        return EXIT_FAIL
    print("check PASS")
    return EXIT_OK


def cmd_demo(args) -> int:
    import demo as demo_mod
    try:
        demo_mod.main()
    except AssertionError as exc:
        print(f"demo FAILED: {exc}", file=sys.stderr)
        return EXIT_FAIL
    return EXIT_OK


# ------------------------------------------------------------------- serve

class _Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: Any) -> None:
        body = (canonical(payload) + "\n").encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        body = json.loads(raw) if raw.strip() else {}
        if not isinstance(body, dict):
            raise ValueError("body must be a JSON object")
        return body

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            return self._send_json(200, {"status": "ok"})
        return self._send_json(404, {"error": "unknown endpoint"})

    def do_POST(self):  # noqa: N802
        try:
            if self.path == "/score":
                body = self._read_body()
                signals = body.get("signals", body)
                return self._send_json(200, score_one(signals))
            return self._send_json(404, {"error": "unknown endpoint"})
        except (ValueError, TypeError, KeyError) as exc:
            return self._send_json(400, {"error": f"bad request: {exc}"})

    def log_message(self, *a: Any) -> None:
        pass


def make_server(port: int) -> HTTPServer:
    return HTTPServer(("127.0.0.1", port), _Handler)


def cmd_serve(args) -> int:
    server = make_server(args.port)
    print(f"serving on http://127.0.0.1:{args.port}  "
          f"(POST /score, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gtm-signal-scoring-eval-kit",
        description="Backtest GTM intent-signal scoring rules on "
                    "synthetic accounts; catch signal decay and "
                    "timestamp leakage (synthetic sample data only).")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a synthetic dataset")
    g.add_argument("--out", required=True)
    g.add_argument("--truth", default=None,
                   help="also write the ground-truth JSON here")
    g.add_argument("--seed", type=int, default=DEFAULT_SEED)
    g.add_argument("--variant", choices=["seeded", "clean"],
                   default="seeded")
    g.set_defaults(func=cmd_generate)

    for name, help_text, func in (
            ("signals", "compute the 5 signals + per-window metrics",
             cmd_signals),
            ("score", "combined score, weights and AUCs", cmd_score),
            ("report", "full report: metrics, flags, weights, AUCs",
             cmd_report),
            ("check", "gate: exit 0 clean / 1 leakage or decay / 2 error",
             cmd_check)):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--input", default=None,
                        help="dataset JSON (default: seeded generator)")
        sp.add_argument("--out", default=None)
        sp.set_defaults(func=func)

    sub.add_parser("demo", help="run the end-to-end demo").set_defaults(
        func=cmd_demo)

    s = sub.add_parser("serve", help="localhost JSON API")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
