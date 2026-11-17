import math
import unittest

from generator import SIGNAL_NAMES
from scoring import auc, combined_score, compute_weights, score_dataset
from signals import all_metrics, compute_signals, decay_check, flags_from, \
    leakage_check
from tests.helpers import seeded


class TestAuc(unittest.TestCase):
    def test_perfect_separation(self):
        self.assertEqual(auc([(0.9, True), (0.8, True),
                              (0.2, False), (0.1, False)]), 1.0)

    def test_reversed_separation(self):
        self.assertEqual(auc([(0.1, True), (0.2, True),
                              (0.8, False), (0.9, False)]), 0.0)

    def test_all_ties_is_half(self):
        self.assertEqual(auc([(0.5, True), (0.5, False)]), 0.5)

    def test_hand_computed_with_ties(self):
        # positives: 0.7, 0.4 ; negatives: 0.4, 0.1
        # pairs: (0.7>0.4)=1, (0.7>0.1)=1, (0.4=0.4)=.5, (0.4>0.1)=1
        # AUC = 3.5/4 = 0.875
        self.assertEqual(auc([(0.7, True), (0.4, True),
                              (0.4, False), (0.1, False)]), 0.875)

    def test_auc_rejects_single_class(self):
        with self.assertRaises(ValueError):
            auc([(0.5, True), (0.6, True)])
        with self.assertRaises(ValueError):
            auc([])


class TestWeights(unittest.TestCase):
    def _fit_metrics(self, lifts):
        return {s: {"lift": lifts[s]} for s in SIGNAL_NAMES}

    def test_weight_formula_is_log_of_fit_lift(self):
        lifts = {s: 2.0 for s in SIGNAL_NAMES}
        w = compute_weights(self._fit_metrics(lifts), excluded=[])
        for s in SIGNAL_NAMES:
            self.assertAlmostEqual(w[s], math.log(2.0))

    def test_excluded_signal_gets_zero_weight(self):
        lifts = {s: 5.0 for s in SIGNAL_NAMES}
        w = compute_weights(self._fit_metrics(lifts),
                            excluded=["contract_doc_download"])
        self.assertEqual(w["contract_doc_download"], 0.0)
        self.assertGreater(w["webinar_attendance"], 0.0)

    def test_lift_at_or_below_one_gets_zero_weight(self):
        lifts = {s: 2.0 for s in SIGNAL_NAMES}
        lifts["webinar_attendance"] = 1.0
        lifts["pricing_page_visits"] = 0.8
        w = compute_weights(self._fit_metrics(lifts), excluded=[])
        self.assertEqual(w["webinar_attendance"], 0.0)
        self.assertEqual(w["pricing_page_visits"], 0.0)

    def test_combined_score_sums_fired_weights_only(self):
        weights = {s: 1.0 for s in SIGNAL_NAMES}
        firing = {s: False for s in SIGNAL_NAMES}
        firing["pricing_page_visits"] = True
        firing["webinar_attendance"] = True
        self.assertEqual(combined_score(firing, weights), 2.0)
        self.assertEqual(combined_score(
            {s: False for s in SIGNAL_NAMES}, weights), 0.0)


class TestSeededScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = seeded()
        cls.comp = compute_signals(cls.ds)
        cls.metrics = all_metrics(cls.ds, cls.comp)
        cls.flags = flags_from(leakage_check(cls.ds),
                               decay_check(cls.metrics))
        fit = {s: cls.metrics[s]["fit"] for s in SIGNAL_NAMES}
        cls.weights = compute_weights(fit, excluded=cls.flags["leakage"])
        cls.scored = score_dataset(cls.ds, cls.comp, cls.weights,
                                   excluded=cls.flags["leakage"])

    def test_seeded_weights_match_formula_from_fit_lifts(self):
        for s in SIGNAL_NAMES:
            lift = self.metrics[s]["fit"]["lift"]
            expected = 0.0 if s in self.flags["leakage"] or lift <= 1 \
                else math.log(lift)
            self.assertAlmostEqual(self.weights[s], expected, places=12)

    def test_heldout_auc_beats_single_signal_baseline(self):
        held = self.scored["heldout"]["auc"]
        self.assertGreater(held, self.scored["baseline"]["heldout_auc"])

    def test_aucs_in_sane_band(self):
        self.assertTrue(0.70 < self.scored["fit"]["auc"] < 0.97)
        self.assertTrue(0.65 < self.scored["heldout"]["auc"] < 0.95)

    def test_baseline_is_not_the_leakage_signal(self):
        self.assertNotEqual(self.scored["baseline"]["signal"],
                            "contract_doc_download")

    def test_leakage_signal_alone_would_look_better_and_is_not_used(self):
        # The trap, pinned: contract_doc_download alone beats the
        # honest combined score in-sample. That is why the check
        # exists — and why its weight is still exactly 0.
        single = self.scored["fit"]["single_signal_aucs"]
        self.assertGreater(single["contract_doc_download"],
                           single["champion_engagement"])
        self.assertEqual(self.weights["contract_doc_download"], 0.0)

    def test_score_rows_cover_cohorts(self):
        self.assertEqual(len(self.scored["fit"]["scores"]), 600)
        self.assertEqual(len(self.scored["heldout"]["scores"]), 510)


if __name__ == "__main__":
    unittest.main()
