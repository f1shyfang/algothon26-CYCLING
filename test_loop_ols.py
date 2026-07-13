"""Unit tests for loop.py overlay helpers and production-floor invariance."""
import unittest

import numpy as np

from loop import (
    Params,
    evaluate,
    is_promotable,
    load_prices,
    rolling_ols_beta,
    simulate,
    simulate_range,
    spread_z,
    VOLATILITY_FLOOR,
)


class TestRollingOlsBeta(unittest.TestCase):
    def test_known_slope_no_intercept(self):
        x = np.arange(10, dtype=float)
        y = 2.5 * x
        beta = rolling_ols_beta(y, x, lb=10, intercept=False)
        self.assertAlmostEqual(beta, 2.5, places=6)

    def test_known_slope_with_intercept(self):
        x = np.arange(10, dtype=float)
        y = 3.0 + 1.5 * x
        beta = rolling_ols_beta(y, x, lb=10, intercept=True)
        self.assertAlmostEqual(beta, 1.5, places=5)

    def test_uses_only_last_lb(self):
        x = np.concatenate([np.zeros(5), np.arange(5, dtype=float)])
        y = np.concatenate([np.ones(5) * 100, 2.0 * np.arange(5, dtype=float)])
        beta = rolling_ols_beta(y, x, lb=5, intercept=False)
        self.assertAlmostEqual(beta, 2.0, places=6)


class TestSpreadZ(unittest.TestCase):
    def test_zero_at_mean(self):
        s = np.ones(20)
        self.assertAlmostEqual(spread_z(s, 20), 0.0, places=6)

    def test_positive_when_last_high(self):
        s = np.zeros(20)
        s[-1] = 5.0
        z = spread_z(s, 20)
        self.assertGreater(z, 0.0)


class TestMinimalCoreFloor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prc = load_prices()
        cls.base = evaluate(cls.prc, Params())

    def test_production_floor_fields(self):
        p = Params()
        self.assertEqual(p.ols_weight, 0.20)
        self.assertEqual(p.mpairs_lookback, 40)
        self.assertEqual(p.mpairs_weight, 0.20)
        self.assertEqual(p.mpairs_top_k, 3)
        self.assertEqual(p.mpairs_entry_z, 1.5)
        self.assertEqual(p.mpairs_min_corr, 0.65)
        self.assertEqual(p.pairs_weight, 0.0)
        self.assertEqual(p.algo_signal_scale, 1.0)
        self.assertEqual(p.xs_weight, 0.0)
        self.assertEqual(p.momentum_weight, 0.0)
        self.assertEqual(p.vol_target_lookback, 20)
        self.assertEqual(p.vol_target_floor, 0.70)
        self.assertEqual(p.vol_target_cap, 2.0)

    def test_zero_ols_lookback_change_is_noop(self):
        a = evaluate(self.prc, Params(ols_weight=0.0))
        b = evaluate(self.prc, Params(ols_weight=0.0, ols_lookback=60))
        self.assertAlmostEqual(a.score, b.score, places=2)

    def test_zero_mpairs_is_noop(self):
        a = evaluate(self.prc, Params(mpairs_weight=0.0))
        b = evaluate(self.prc, Params(mpairs_weight=0.0, mpairs_lookback=80))
        self.assertAlmostEqual(a.score, b.score, places=2)

    def test_inactive_mpairs_does_not_dilute(self):
        base = evaluate(self.prc, Params(mpairs_weight=0.0))
        cand = evaluate(
            self.prc,
            Params(mpairs_weight=0.10, mpairs_min_corr=0.85, mpairs_lookback=40),
        )
        self.assertAlmostEqual(cand.score, base.score, places=2)

    def test_floor_score_matches_production(self):
        self.assertAlmostEqual(self.base.score, 249.91, places=2)


class TestSimulateRange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prc = load_prices()
        cls.nt = cls.prc.shape[1]

    def test_official_range_matches_simulate(self):
        p = Params()
        a = simulate(self.prc, p, 250)
        # last 250 days: start_day = nt-250, end_day = nt
        b = simulate_range(self.prc, p, start_day=self.nt - 250, end_day=self.nt)
        self.assertAlmostEqual(a.score, b.score, places=4)
        self.assertEqual(len(a.pll), len(b.pll))
        np.testing.assert_allclose(a.pll, b.pll, rtol=1e-9, atol=1e-9)

    def test_fold_length_is_100(self):
        p = Params()
        # F1: days 201-300 → start_day=200, end_day=300
        r = simulate_range(self.prc, p, start_day=200, end_day=300)
        self.assertEqual(len(r.pll), 100)


class TestWalkForwardPromote(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prc = load_prices()

    def test_evaluate_exposes_three_folds(self):
        e = evaluate(self.prc, Params())
        self.assertTrue(hasattr(e, "fold1_score"))
        self.assertTrue(hasattr(e, "fold2_score"))
        self.assertTrue(hasattr(e, "fold3_score"))
        self.assertIsInstance(e.robust, bool)

    def test_baseline_not_promotable_against_itself(self):
        base = evaluate(self.prc, Params())
        self.assertFalse(is_promotable(base, base))

    def test_promotable_requires_majority_and_official_and_f3(self):
        from dataclasses import replace

        base = evaluate(self.prc, Params())
        good = replace(
            base,
            score=base.score + 1.0,
            fold1_score=base.fold1_score + 1.0,
            fold2_score=base.fold2_score + 1.0,
            fold3_score=base.fold3_score,
            robust=True,
        )
        self.assertTrue(is_promotable(base, good))

        weak = replace(
            base,
            score=base.score + 5.0,
            fold1_score=base.fold1_score + 1.0,
            fold2_score=base.fold2_score - 1.0,
            fold3_score=base.fold3_score - 1.0,
            robust=False,
        )
        self.assertFalse(is_promotable(base, weak))


if __name__ == "__main__":
    unittest.main()
