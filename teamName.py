import numpy as np

LOOKBACKS = (5, 20)
DEFAULT_DOLLAR_LIMIT = 10_000
ALGO_DOLLAR_LIMIT = 100_000
VOLATILITY_FLOOR = 1e-12
REBALANCE_BAND = 0.195
REGIME_VOL_SHORT = 10
REGIME_VOL_LONG = 60
REGIME_THRESHOLD = 1.15
REGIME_SCALE = 0.22
# Track B: ALGO-vs-basket pairs overlay (promoted from loop.py sweep)
PAIRS_LOOKBACK = 40
PAIRS_WEIGHT = 0.20
PAIRS_ENTRY_Z = 2.0
# Track C: ALGO-specific exposure multiplier (promoted from loop.py sweep)
ALGO_SIGNAL_SCALE = 3.0

_current_positions = np.zeros(0, dtype=int)
_last_history = np.zeros((0, 0))


def getMyPosition(prcSoFar):
    global _current_positions, _last_history

    nins, nt = prcSoFar.shape
    longest_lookback = max(max(LOOKBACKS), REGIME_VOL_LONG, PAIRS_LOOKBACK)

    same_history = _last_history.shape == prcSoFar.shape and np.array_equal(
        _last_history,
        prcSoFar,
    )
    if same_history:
        return _current_positions.copy()

    is_continuation = (
        _current_positions.size == nins
        and _last_history.shape == (nins, nt - 1)
        and np.array_equal(_last_history, prcSoFar[:, :-1])
    )
    if not is_continuation:
        _current_positions = np.zeros(nins, dtype=int)

    if nt <= longest_lookback:
        _last_history = prcSoFar.copy()
        return _current_positions.copy()

    recent_prices = prcSoFar[:, -(longest_lookback + 1) :]
    log_prices = np.log(recent_prices)
    daily_returns = np.diff(log_prices, axis=1)

    signal = np.zeros(nins)
    for lookback in LOOKBACKS:
        cumulative_return = log_prices[:, -1] - log_prices[:, -(lookback + 1)]
        volatility = daily_returns[:, -lookback:].std(axis=1) * np.sqrt(lookback)
        standardized_return = cumulative_return / np.maximum(
            volatility, VOLATILITY_FLOOR
        )
        signal -= np.clip(standardized_return, -1.0, 1.0) / len(LOOKBACKS)

    # Track B: ALGO-vs-basket residual (equal-weight basket of instruments 1..n)
    basket = np.nanmean(log_prices[1:, :], axis=0)
    algo_log_price = log_prices[0, :]
    spread = algo_log_price - basket
    spread_mean = spread[-PAIRS_LOOKBACK:].mean()
    spread_std = spread[-PAIRS_LOOKBACK:].std() + VOLATILITY_FLOOR
    z = (spread[-1] - spread_mean) / spread_std
    if abs(z) >= PAIRS_ENTRY_Z:
        pair_signal = np.zeros(nins)
        pair_signal[0] = -np.clip(z, -1.0, 1.0)
        pair_signal[1:] = -pair_signal[0] / (nins - 1)
        signal = (1.0 - PAIRS_WEIGHT) * signal + PAIRS_WEIGHT * pair_signal

    # Track C: scale up ALGO's own exposure relative to the rest of the basket
    signal[0] *= ALGO_SIGNAL_SCALE

    short_vol = daily_returns[:, -REGIME_VOL_SHORT:].std(axis=1)
    long_vol = daily_returns[:, -REGIME_VOL_LONG:].std(axis=1)
    vol_ratio = short_vol / np.maximum(long_vol, VOLATILITY_FLOOR)
    exposure = np.where(vol_ratio >= REGIME_THRESHOLD, REGIME_SCALE, 1.0)

    dollar_limits = np.full(nins, DEFAULT_DOLLAR_LIMIT, dtype=float)
    dollar_limits[0] = ALGO_DOLLAR_LIMIT
    target_dollars = dollar_limits * np.clip(signal, -1.0, 1.0) * exposure
    current_prices = prcSoFar[:, -1]
    desired_positions = np.trunc(target_dollars / current_prices).astype(int)

    rebalance_notional = np.abs(desired_positions - _current_positions) * current_prices
    should_rebalance = rebalance_notional >= REBALANCE_BAND * dollar_limits
    new_positions = np.where(
        should_rebalance,
        desired_positions,
        _current_positions,
    )

    max_shares = np.trunc(dollar_limits / current_prices).astype(int)
    _current_positions = np.clip(new_positions, -max_shares, max_shares)
    _last_history = prcSoFar.copy()
    return _current_positions.copy()
