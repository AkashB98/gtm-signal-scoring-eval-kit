import io
import json
import os
import tempfile
import threading
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout

from cli import EXIT_ERROR, EXIT_FAIL, EXIT_OK, main as cli_main, \
    make_server, score_one
from generator import SIGNAL_NAMES, generate_dataset
from report import build_report
from validation import validate_dataset, validate_score_request
from tests.helpers import seeded


def _run(argv):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return cli_main(argv)


class TestCliCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="gtm-signal-test-")
        cls.seeded_path = os.path.join(cls.tmp, "seeded.json")
        cls.clean_path = os.path.join(cls.tmp, "clean.json")
        with open(cls.seeded_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(generate_dataset(), sort_keys=True) + "\n")
        with open(cls.clean_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(generate_dataset(variant="clean"),
                                sort_keys=True) + "\n")

    def test_check_seeded_exits_1_by_design(self):
        self.assertEqual(_run(["check", "--input", self.seeded_path]),
                         EXIT_FAIL)

    def test_check_clean_exits_0(self):
        self.assertEqual(_run(["check", "--input", self.clean_path]),
                         EXIT_OK)

    def test_check_default_input_is_seeded_and_exits_1(self):
        self.assertEqual(_run(["check"]), EXIT_FAIL)

    def test_check_missing_file_exits_2(self):
        self.assertEqual(_run(["check", "--input", "/tmp/no-ds-xyz.json"]),
                         EXIT_ERROR)

    def test_check_malformed_json_exits_2(self):
        bad = os.path.join(self.tmp, "bad.json")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        self.assertEqual(_run(["check", "--input", bad]), EXIT_ERROR)

    def test_check_invalid_dataset_exits_2(self):
        bad = os.path.join(self.tmp, "invalid.json")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"accounts": []}) + "\n")
        self.assertEqual(_run(["check", "--input", bad]), EXIT_ERROR)

    def test_generate_writes_dataset_and_truth(self):
        out = os.path.join(self.tmp, "gen.json")
        truth = os.path.join(self.tmp, "truth.json")
        self.assertEqual(_run(["generate", "--out", out,
                               "--truth", truth]), EXIT_OK)
        with open(out, encoding="utf-8") as fh:
            ds = json.load(fh)
        self.assertTrue(validate_dataset(ds))
        with open(truth, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["counts"]["accounts"], 600)

    def test_generate_bad_variant_exits_2(self):
        # argparse rejects unknown variants with exit code 2 semantics;
        # the generate path itself raises ValueError -> EXIT_ERROR.
        out = os.path.join(self.tmp, "x.json")
        code = None
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                code = cli_main(["generate", "--out", out,
                                 "--variant", "bogus"])
            except SystemExit as exc:
                code = exc.code
        self.assertEqual(code, 2)

    def test_report_command_matches_build_report(self):
        out = os.path.join(self.tmp, "report.json")
        self.assertEqual(_run(["report", "--input", self.seeded_path,
                               "--out", out]), EXIT_OK)
        with open(out, encoding="utf-8") as fh:
            payload = json.load(fh)
        self.assertEqual(payload, build_report(seeded()))

    def test_signals_and_score_commands(self):
        out = os.path.join(self.tmp, "signals.json")
        self.assertEqual(_run(["signals", "--input", self.seeded_path,
                               "--out", out]), EXIT_OK)
        with open(out, encoding="utf-8") as fh:
            payload = json.load(fh)
        self.assertEqual(len(payload["signals"]), 600)
        out2 = os.path.join(self.tmp, "score.json")
        self.assertEqual(_run(["score", "--input", self.seeded_path,
                               "--out", out2]), EXIT_OK)
        with open(out2, encoding="utf-8") as fh:
            scored = json.load(fh)
        self.assertEqual(scored["excluded_from_score"],
                         ["contract_doc_download"])
        self.assertEqual(len(scored["fit_scores"]), 600)
        self.assertEqual(len(scored["heldout_scores"]), 510)

    def test_report_determinism(self):
        self.assertEqual(build_report(seeded()), build_report(seeded()))


class TestServeAndValidation(unittest.TestCase):
    def _request(self, port, path, payload=None, method="GET"):
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     data=data, headers=headers,
                                     method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_serve_health_score_and_errors(self):
        server = make_server(0)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            status, health = self._request(port, "/health")
            self.assertEqual((status, health), (200, {"status": "ok"}))
            sample = {s: False for s in SIGNAL_NAMES}
            sample["pricing_page_visits"] = True
            status, served = self._request(
                port, "/score", payload={"signals": sample}, method="POST")
            self.assertEqual(status, 200)
            self.assertEqual(served, score_one(sample))
            status, _body = self._request(
                port, "/score", payload={"signals": {"nope": True}},
                method="POST")
            self.assertEqual(status, 400)
            status, _body = self._request(port, "/nope")
            self.assertEqual(status, 404)
        finally:
            server.shutdown()
            server.server_close()

    def test_score_one_excludes_leakage_contribution(self):
        sample = {s: False for s in SIGNAL_NAMES}
        sample["contract_doc_download"] = True
        result = score_one(sample)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["contributions"]
                         ["contract_doc_download"], 0.0)
        self.assertEqual(result["excluded_signals"],
                         ["contract_doc_download"])

    def test_validate_score_request_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            validate_score_request({"signals": {"pricing_page_visits": 1}})
        with self.assertRaises(ValueError):
            validate_score_request({"signals": {}})
        with self.assertRaises(ValueError):
            validate_score_request(["not", "a", "dict"])
        good = validate_score_request(
            {"signals": {s: True for s in SIGNAL_NAMES}})
        self.assertEqual(set(good), set(SIGNAL_NAMES))

    def test_validate_dataset_rejects_duplicates_and_bad_conversion(self):
        ds = generate_dataset()
        self.assertTrue(validate_dataset(ds))
        dup = json.loads(json.dumps(ds))
        dup["accounts"].append(dict(dup["accounts"][0]))
        with self.assertRaises(ValueError):
            validate_dataset(dup)
        bad = json.loads(json.dumps(ds))
        bad["accounts"][0]["conversion_day"] = 999
        with self.assertRaises(ValueError):
            validate_dataset(bad)


class TestNoNetwork(unittest.TestCase):
    def test_report_builds_with_sockets_blocked(self):
        import socket

        original = socket.socket

        def _blocked(*a, **k):
            raise AssertionError("network use is forbidden in core logic")

        socket.socket = _blocked
        try:
            report = build_report(seeded())
        finally:
            socket.socket = original
        self.assertEqual(report["accounts"], 600)


if __name__ == "__main__":
    unittest.main()
