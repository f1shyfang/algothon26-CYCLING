import unittest

import numpy as np

from teamName import getMyPosition


N_INSTRUMENTS = 51
# Must exceed the production lead-lag lookback so signals are active.
MIN_HISTORY = 340


def make_leadlag_prices(final_leader_return):
    rng = np.random.default_rng(26)
    days = 360
    returns = rng.normal(0.0, 0.002, (N_INSTRUMENTS, days - 1))
    leader_returns = rng.normal(0.0, 0.015, days - 2)
    returns[0, :-1] = leader_returns
    returns[1, 1:] = 0.8 * leader_returns + rng.normal(0.0, 0.001, days - 2)
    returns[0, -1] = final_leader_return
    log_prices = np.concatenate(
        [np.zeros((N_INSTRUMENTS, 1)), np.cumsum(returns, axis=1)],
        axis=1,
    )
    return 100.0 * np.exp(log_prices)


class PositionTests(unittest.TestCase):
    def test_waits_for_longest_lookback(self):
        prices = np.full((N_INSTRUMENTS, 20), 100.0)

        positions = getMyPosition(prices)

        np.testing.assert_array_equal(positions, np.zeros(N_INSTRUMENTS, dtype=int))

    def test_positive_leader_return_produces_long_follower_position(self):
        prices = make_leadlag_prices(0.08)

        positions = getMyPosition(prices)

        self.assertGreater(positions[1], 0)
        self.assertLessEqual(abs(positions[0] * prices[0, -1]), 100_000)
        self.assertTrue(
            np.all(np.abs(positions[1:] * prices[1:, -1]) <= 10_000)
        )

    def test_negative_leader_return_produces_short_follower_position(self):
        prices = make_leadlag_prices(-0.08)

        positions = getMyPosition(prices)

        self.assertLess(positions[1], 0)

    def test_vol_targeting_keeps_returned_positions_within_official_limits(self):
        # Instrument 1 has much lower volatility than the rest, which drives
        # its inverse-vol allocation to the configured cap.
        days = MIN_HISTORY
        prices = np.full((N_INSTRUMENTS, days), 100.0)
        prices[0] = 100.0 * np.exp(np.linspace(0.0, 0.2, days))
        prices[1] = 100.0 * np.exp(np.linspace(0.0, 0.02, days))
        for instrument in range(2, N_INSTRUMENTS):
            prices[instrument] = 100.0 * np.exp(
                np.linspace(0.0, 0.2, days) + 0.01 * np.sin(np.arange(days))
            )

        positions = getMyPosition(prices)
        limits = np.full(N_INSTRUMENTS, 10_000.0)
        limits[0] = 100_000.0
        self.assertTrue(np.all(np.abs(positions * prices[:, -1]) <= limits))

    def test_constant_prices_produce_zero_positions(self):
        prices = np.full((N_INSTRUMENTS, MIN_HISTORY), 100.0)

        positions = getMyPosition(prices)

        np.testing.assert_array_equal(positions, np.zeros(N_INSTRUMENTS, dtype=int))

    def test_same_history_always_produces_same_target(self):
        prices = np.exp(np.linspace(0.0, 0.1, MIN_HISTORY))[None, :]
        prices = np.repeat(prices, N_INSTRUMENTS, axis=0) * 100.0

        first = getMyPosition(prices)
        second = getMyPosition(prices)

        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.issubdtype(first.dtype, np.integer))

    def test_new_leader_return_updates_positions_within_limits(self):
        prices = make_leadlag_prices(0.08)
        first = getMyPosition(prices)

        # A fresh leader impulse is legitimate new information for the
        # lead-lag strategy, so the desired follower set may change.
        next_returns = np.zeros(N_INSTRUMENTS)
        next_returns[0] = 0.079
        next_returns[1] = 0.063
        next_prices = np.concatenate(
            [prices, prices[:, -1:] * np.exp(next_returns[:, None])],
            axis=1,
        )
        second = getMyPosition(next_prices)

        self.assertFalse(np.array_equal(second, first))
        limits = np.full(N_INSTRUMENTS, 10_000.0)
        limits[0] = 100_000.0
        self.assertTrue(np.all(np.abs(second * next_prices[:, -1]) <= limits))

    def test_repeated_history_preserves_band_retained_position(self):
        rng = np.random.default_rng(7)
        returns = rng.normal(0.001, 0.01, MIN_HISTORY - 1)
        prices = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(returns))))
        prices = np.repeat(prices[None, :], N_INSTRUMENTS, axis=0)
        getMyPosition(prices)
        next_prices = np.concatenate(
            [prices, prices[:, -1:] * np.exp(-0.002)],
            axis=1,
        )
        retained = getMyPosition(next_prices)

        repeated = getMyPosition(next_prices)

        np.testing.assert_array_equal(repeated, retained)

    def test_unrelated_longer_history_does_not_inherit_positions(self):
        first_prices = np.exp(np.linspace(0.0, 0.1, MIN_HISTORY))[None, :] * 100.0
        first_prices = np.repeat(first_prices, N_INSTRUMENTS, axis=0)
        getMyPosition(first_prices)
        unrelated = np.exp(np.linspace(0.0, np.log(1.05), MIN_HISTORY + 2))[None, :] * 100.0
        unrelated = np.repeat(unrelated, N_INSTRUMENTS, axis=0)

        first_call = getMyPosition(unrelated)
        repeated_call = getMyPosition(unrelated)

        np.testing.assert_array_equal(first_call, repeated_call)


if __name__ == "__main__":
    unittest.main()
