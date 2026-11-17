import unittest

from generator import SIGNAL_NAMES
from signals import (all_metrics, cohort_and_label, compute_signals,
                     decay_check, flags_from, leakage_check, signal_metrics)
from tests.helpers import account, clean, seeded, tiny_dataset


def _ev(account_id, event_type, day):
    return {"account_id": account_id, "event_type": event_type,
            "day": day, "date": "2026-01-05", "week": day // 7}


class TestThresholds(unittest.TestCase):
    def _ds(self):
        accounts = [account("X1"), account("X2"), account("X3")]
        weekly = [
            {"account_id": "X1", "week": 2, "sessions": 6,
             "active_users": 3},
            {"account_id": "X2", "week": 2, "sessions": 5,
             "active_users": 3},
            {"account_id": "X3", "week": 2, "sessions": 9,
             "active_users": 3},
        ]
        events = (
            [_ev("X1", "pricing_page_view", d) for d in (1, 2, 3)] +
            [_ev("X2", "pricing_page_view", d) for d in (1, 2)] +
            [_ev("X1", "champion_engagement", d) for d in (4, 5)] +
            [_ev("X2", "champion_engagement", 4)] +
            [_ev("X3", "webinar_attendance", 6)] +
            [_ev("X3", "contract_doc_download", 7)]
        )
        return tiny_dataset(accounts, weekly, events)

    def test_spike_threshold_is_six_sessions(self):
        comp = compute_signals(self._ds())
        self.assertTrue(comp["X1"]["fit"]["product_usage_spike"])
        self.assertFalse(comp["X2"]["fit"]["product_usage_spike"])
        self.assertTrue(comp["X3"]["fit"]["product_usage_spike"])

    def test_pricing_threshold_is_three(self):
        comp = compute_signals(self._ds())
        self.assertTrue(comp["X1"]["fit"]["pricing_page_visits"])
        self.assertFalse(comp["X2"]["fit"]["pricing_page_visits"])

    def test_champion_threshold_is_two(self):
        comp = compute_signals(self._ds())
        self.assertTrue(comp["X1"]["fit"]["champion_engagement"])
        self.assertFalse(comp["X2"]["fit"]["champion_engagement"])

    def test_webinar_and_contract_threshold_is_one(self):
        comp = compute_signals(self._ds())
        self.assertTrue(comp["X3"]["fit"]["webinar_attendance"])
        self.assertTrue(comp["X3"]["fit"]["contract_doc_download"])
        self.assertFalse(comp["X1"]["fit"]["webinar_attendance"])

    def test_events_outside_window_do_not_fire_it(self):
        comp = compute_signals(self._ds())
        for signal in SIGNAL_NAMES:
            self.assertFalse(comp["X3"]["heldout"][signal])


class TestSeededSignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = seeded()
        cls.comp = compute_signals(cls.ds)
        cls.metrics = all_metrics(cls.ds, cls.comp)
        cls.leakage = leakage_check(cls.ds)
        cls.decay = decay_check(cls.metrics)

    def test_recompute_matches_planted_everywhere(self):
        planted = self.ds["ground_truth"]["planted_firing"]
        for a in planted:
            for w in planted[a]:
                self.assertEqual(planted[a][w], self.comp[a][w], (a, w))

    def test_leakage_flagged_exactly_contract(self):
        flags = flags_from(self.leakage, self.decay)
        self.assertEqual(flags["leakage"], ["contract_doc_download"])
        offending = self.leakage["contract_doc_download"][
            "post_conversion_timestamps"]
        self.assertGreater(offending, 100)

    def test_honest_and_decay_signals_have_zero_leakage(self):
        for s in SIGNAL_NAMES:
            if s != "contract_doc_download":
                self.assertEqual(
                    self.leakage[s]["post_conversion_timestamps"], 0, s)
                self.assertFalse(self.leakage[s]["flagged_leakage"], s)

    def test_decay_flagged_exactly_webinar(self):
        flags = flags_from(self.leakage, self.decay)
        self.assertEqual(flags["decay"], ["webinar_attendance"])
        ratio = self.decay["webinar_attendance"]["heldout_to_fit_ratio"]
        self.assertLess(ratio, 0.40)

    def test_honest_signals_flagged_by_neither(self):
        for s in ("product_usage_spike", "pricing_page_visits",
                  "champion_engagement"):
            self.assertFalse(self.leakage[s]["flagged_leakage"], s)
            self.assertFalse(self.decay[s]["flagged_decay"], s)

    def test_honest_lifts_above_two_in_both_windows(self):
        for s in ("product_usage_spike", "pricing_page_visits",
                  "champion_engagement"):
            self.assertGreater(self.metrics[s]["fit"]["lift"], 2.0, s)
            self.assertGreater(self.metrics[s]["heldout"]["lift"], 2.0, s)

    def test_leakage_signal_looks_near_perfect_in_sample(self):
        m = self.metrics["contract_doc_download"]["fit"]
        self.assertGreater(m["precision"], 0.8)
        self.assertGreater(m["lift"], 4.0)

    def test_cohorts_exclude_already_converted(self):
        fit_ids, _ = cohort_and_label(self.ds, "fit")
        held_ids, held_labels = cohort_and_label(self.ds, "heldout")
        self.assertEqual(len(fit_ids), 600)
        self.assertEqual(len(held_ids), 510)
        self.assertEqual(sum(held_labels.values()), 90)

    def test_metric_counts_sum_to_cohort(self):
        for s in SIGNAL_NAMES:
            for w in ("fit", "heldout"):
                m = self.metrics[s][w]
                self.assertEqual(m["tp"] + m["fp"] + m["fn"] + m["tn"],
                                 m["cohort"], (s, w))
                self.assertEqual(m["tp"] + m["fn"], m["positives"])


class TestHandComputedMetrics(unittest.TestCase):
    def test_precision_recall_lift_on_tiny_set(self):
        # 4 accounts in fit cohort, 2 convert. Signal fires for one
        # converter and one non-converter: precision .5, recall .5,
        # base .5, lift 1.0.
        accounts = [account("A", 20), account("B", 30),
                    account("C"), account("D")]
        ds = tiny_dataset(accounts)
        comp = {a: {"fit": {s: False for s in SIGNAL_NAMES},
                    "heldout": {s: False for s in SIGNAL_NAMES}}
                for a in ("A", "B", "C", "D")}
        comp["A"]["fit"]["webinar_attendance"] = True
        comp["C"]["fit"]["webinar_attendance"] = True
        m = signal_metrics(ds, comp, "fit", "webinar_attendance")
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (1, 1, 1, 1))
        self.assertEqual(m["precision"], 0.5)
        self.assertEqual(m["recall"], 0.5)
        self.assertEqual(m["base_rate"], 0.5)
        self.assertEqual(m["lift"], 1.0)

    def test_leakage_check_on_tiny_set(self):
        accounts = [account("A", 20), account("B")]
        events = [_ev("A", "contract_doc_download", 22),
                  _ev("A", "pricing_page_view", 5),
                  _ev("B", "contract_doc_download", 9)]
        ds = tiny_dataset(accounts, [], events)
        leak = leakage_check(ds)
        self.assertTrue(leak["contract_doc_download"]["flagged_leakage"])
        self.assertEqual(leak["contract_doc_download"]
                         ["post_conversion_timestamps"], 1)
        self.assertFalse(leak["pricing_page_visits"]["flagged_leakage"])

    def test_clean_variant_flags_nothing(self):
        ds = clean()
        comp = compute_signals(ds)
        flags = flags_from(leakage_check(ds),
                           decay_check(all_metrics(ds, comp)))
        self.assertEqual(flags, {"leakage": [], "decay": []})


if __name__ == "__main__":
    unittest.main()
