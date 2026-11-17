import unittest

from generator import SIGNAL_NAMES, day_to_date, generate_dataset
from tests.helpers import clean, seeded


class TestGenerator(unittest.TestCase):
    def test_determinism(self):
        self.assertEqual(generate_dataset(), generate_dataset())

    def test_account_count_and_ids(self):
        ds = seeded()
        ids = [a["account_id"] for a in ds["accounts"]]
        self.assertEqual(len(ids), 600)
        self.assertEqual(len(set(ids)), 600)
        self.assertEqual(ids[0], "A001")
        self.assertEqual(ids[-1], "A600")

    def test_company_names_unique_and_fictional_vendor(self):
        ds = seeded()
        names = [a["company"] for a in ds["accounts"]]
        self.assertEqual(len(set(names)), 600)
        self.assertEqual(ds["meta"]["vendor"], "Meridian Metrics")
        self.assertTrue(ds["meta"]["synthetic"])

    def test_conversion_counts_exact(self):
        counts = seeded()["ground_truth"]["counts"]
        self.assertEqual(counts, {"accounts": 600, "fit_converters": 90,
                                  "heldout_converters": 90,
                                  "never_converters": 420})

    def test_conversion_windows(self):
        for a in seeded()["accounts"]:
            conv = a["conversion_day"]
            if a["account_id"] and conv is not None:
                self.assertIn(a["conversion_week"],
                              list(range(8, 13)) + list(range(20, 26)))
                self.assertEqual(conv, a["conversion_week"] * 7 + 3)
                self.assertEqual(a["conversion_date"], day_to_date(conv))
            self.assertEqual(a["converted"], conv is not None)

    def test_fit_and_heldout_converter_windows_disjoint(self):
        truth = seeded()["ground_truth"]["accounts"]
        fit = {k for k, v in truth.items()
               if v["conversion_window"] == "fit"}
        held = {k for k, v in truth.items()
                if v["conversion_window"] == "heldout"}
        self.assertEqual((len(fit), len(held)), (90, 90))
        self.assertFalse(fit & held)

    def test_weekly_activity_shape(self):
        rows = seeded()["weekly_activity"]
        self.assertEqual(len(rows), 600 * 26)
        self.assertTrue(all(0 <= r["week"] <= 25 for r in rows))
        self.assertTrue(all(0 <= r["sessions"] <= 9 for r in rows))

    def test_events_within_horizon_and_sorted(self):
        events = seeded()["events"]
        self.assertTrue(all(0 <= e["day"] <= 181 for e in events))
        keys = [(e["day"], e["account_id"], e["event_type"])
                for e in events]
        self.assertEqual(keys, sorted(keys))
        self.assertTrue(all(e["week"] == e["day"] // 7 for e in events))
        self.assertTrue(all(e["date"] == day_to_date(e["day"])
                            for e in events))

    def test_no_signal_activity_after_conversion_except_contract(self):
        ds = seeded()
        conv = {a["account_id"]: a["conversion_day"]
                for a in ds["accounts"]}
        for e in ds["events"]:
            c = conv[e["account_id"]]
            if c is not None and e["event_type"] != "contract_doc_download":
                self.assertLessEqual(e["day"], c, e)

    def test_contract_events_for_converters_are_after_conversion(self):
        ds = seeded()
        conv = {a["account_id"]: a["conversion_day"]
                for a in ds["accounts"]}
        after = sum(1 for e in ds["events"]
                    if e["event_type"] == "contract_doc_download"
                    and conv[e["account_id"]] is not None
                    and e["day"] == conv[e["account_id"]] + 2)
        self.assertGreater(after, 150)

    def test_signal_names_exactly_five(self):
        self.assertEqual(SIGNAL_NAMES,
                         ["product_usage_spike", "pricing_page_visits",
                          "champion_engagement", "webinar_attendance",
                          "contract_doc_download"])

    def test_different_seed_changes_dataset(self):
        self.assertNotEqual(generate_dataset(seed=7), seeded())

    def test_bad_variant_raises(self):
        with self.assertRaises(ValueError):
            generate_dataset(variant="bogus")

    def test_clean_variant_same_counts(self):
        self.assertEqual(clean()["ground_truth"]["counts"],
                         seeded()["ground_truth"]["counts"])
        self.assertEqual(clean()["ground_truth"]["planted_patterns"],
                         {"decay_signal": None, "leakage_signal": None})


if __name__ == "__main__":
    unittest.main()
