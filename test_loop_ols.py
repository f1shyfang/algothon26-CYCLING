"""Unit tests for loop.py OLS helpers and default-off invariance."""
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


class TestDefaultOffInvariance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prc = load_prices()

    def test_explicit_zero_ols_is_stable(self):
        # Research-off path: no ols and no pairs
        a = evaluate(self.prc, Params(ols_weight=0.0, pairs_weight=0.0))
        b = evaluate(self.prc, Params(ols_weight=0.0, pairs_weight=0.0, ols_lookback=60))
        self.assertAlmostEqual(a.score, b.score, places=2)

    def test_explicit_zero_mpairs_matches_baseline(self):
        # Production defaults now have ols on; mpairs remains off
        base = evaluate(self.prc, Params())
        cand = evaluate(self.prc, Params(mpairs_weight=0.0, mpairs_lookback=80))
        self.assertAlmostEqual(cand.score, base.score, places=2)

    def test_inactive_mpairs_does_not_dilute_baseline(self):
        # High min_corr selects no pairs on this dataset; gated blend must not dilute.
        base = evaluate(self.prc, Params())
        cand = evaluate(
            self.prc,
            Params(mpairs_weight=0.10, mpairs_min_corr=0.85, mpairs_lookback=40),
        )
        self.assertAlmostEqual(cand.score, base.score, places=2)

    def test_production_baseline_score(self):
        base = evaluate(self.prc, Params())
        self.assertAlmostEqual(base.score, 265.18, places=2)


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


if __name__ == "__main__":
    unittest.main()
