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
MPAIRS_LOOKBACK = 40
MPAIRS_WEIGHT = 0.20
MPAIRS_TOP_K = 3
MPAIRS_ENTRY_Z = 1.5
MPAIRS_MIN_CORR = 0.65
OLS_LOOKBACK = 40
OLS_WEIGHT = 0.20
OLS_ENTRY_Z = 2.0
ADAPTIVE_BAND_VOL_LB = 10
ADAPTIVE_BAND_SCALE = 2.0

_current_positions = np.zeros(0, dtype=int)
_last_history = np.zeros((0, 0))


def _rolling_ols_beta(y, x, lb):
    yy = np.asarray(y[-lb:], dtype=float)
    xx = np.asarray(x[-lb:], dtype=float)
    denom = float(xx @ xx)
    if denom < VOLATILITY_FLOOR:
        return 0.0
    return float(xx @ yy) / denom


def _spread_z(spread, lb):
    window = np.asarray(spread[-lb:], dtype=float)
    mu = float(window.mean())
    sd = float(window.std()) + VOLATILITY_FLOOR
    return (float(window[-1]) - mu) / sd


def _mpairs_signal(log_prices, daily_returns):
    nins = log_prices.shape[0]
    sig = np.zeros(nins)
    rets = daily_returns[1:, -MPAIRS_LOOKBACK:]
    n = rets.shape[0]
    if n < 2 or MPAIRS_LOOKBACK < 3:
        return sig

    # Pairwise abs corr using vectorized corrcoef for speed
    stds = np.std(rets, axis=1)
    valid = stds >= VOLATILITY_FLOOR
    corr_matrix = np.corrcoef(rets)
    candidates = []
    for i in range(n):
        if not valid[i]:
            continue
        for j in range(i + 1, n):
            if not valid[j]:
                continue
            corr = corr_matrix[i, j]
            if np.isnan(corr):
                continue
            if abs(corr) >= MPAIRS_MIN_CORR:
                candidates.append((abs(corr), i + 1, j + 1))
    candidates.sort(reverse=True)

    active = 0
    for _, i, j in candidates[:MPAIRS_TOP_K]:
        beta = _rolling_ols_beta(
            log_prices[i, :],
            log_prices[j, :],
            MPAIRS_LOOKBACK,
        )
        spread = log_prices[i, :] - beta * log_prices[j, :]
        z = _spread_z(spread, MPAIRS_LOOKBACK)
        if abs(z) < MPAIRS_ENTRY_Z:
            continue
        zc = float(np.clip(z, -1.0, 1.0))
        sig[i] += -zc
        sig[j] += zc * beta
        active += 1

    if active == 0:
        return np.zeros(nins)
    sig /= active
    max_abs = np.max(np.abs(sig)) + VOLATILITY_FLOOR
    return np.clip(sig / max_abs, -1.0, 1.0)


def getMyPosition(prcSoFar):
    global _current_positions, _last_history

    nins, nt = prcSoFar.shape
    longest_lookback = max(max(LOOKBACKS), REGIME_VOL_LONG, MPAIRS_LOOKBACK, OLS_LOOKBACK)
    if ADAPTIVE_BAND_VOL_LB > 0:
        longest_lookback = max(longest_lookback, ADAPTIVE_BAND_VOL_LB)

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

    if OLS_WEIGHT > 0 and nt > OLS_LOOKBACK:
        basket = np.nanmean(log_prices[1:, :], axis=0)
        algo = log_prices[0, :]
        beta = _rolling_ols_beta(algo, basket, OLS_LOOKBACK)
        spread = algo - beta * basket
        z = _spread_z(spread, OLS_LOOKBACK)
        if abs(z) >= OLS_ENTRY_Z:
            ols_sig = np.zeros(nins)
            ols_sig[0] = -np.clip(z, -1.0, 1.0)
            ols_sig[1:] = -ols_sig[0] / (nins - 1)
            signal = (1.0 - OLS_WEIGHT) * signal + OLS_WEIGHT * ols_sig

    mpairs_signal = _mpairs_signal(log_prices, daily_returns)
    if np.any(np.abs(mpairs_signal) > VOLATILITY_FLOOR):
        signal = (1.0 - MPAIRS_WEIGHT) * signal + MPAIRS_WEIGHT * mpairs_signal

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
    band = REBALANCE_BAND
    if ADAPTIVE_BAND_VOL_LB > 0 and daily_returns.shape[1] >= 2 * ADAPTIVE_BAND_VOL_LB:
        ab_short = daily_returns[:, -ADAPTIVE_BAND_VOL_LB:].std(axis=1)
        ab_long = daily_returns[:, -2 * ADAPTIVE_BAND_VOL_LB:].std(axis=1)
        ab_ratio = ab_short / np.maximum(ab_long, VOLATILITY_FLOOR)
        if float(np.median(ab_ratio)) >= 1.0:
            band *= ADAPTIVE_BAND_SCALE
    should_rebalance = rebalance_notional >= band * dollar_limits
    new_positions = np.where(
        should_rebalance,
        desired_positions,
        _current_positions,
    )

    max_shares = np.trunc(dollar_limits / current_prices).astype(int)
    _current_positions = np.clip(new_positions, -max_shares, max_shares)
    _last_history = prcSoFar.copy()
    return _current_positions.copy()
