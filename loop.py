#!/usr/bin/env python
"""Automated strategy iteration loop for Algothon 2026 (CYCLING).

This turns the manual process in STRATEGY_LOOP.md into an executable loop:

    1. Simulate a parameterised strategy with the *exact* scoring eval.py uses.
    2. Score it not just on the official 250-day window, but also on a held-out
       "train" window and on both 125-day halves of the test window, so we can
       reject candidates that only win by overfitting the scored window.
    3. Sweep parameter grids, log every result to results.csv, print a
       leaderboard, and surface the best candidate that improves the official
       score *without* regressing the robustness checks.

Nothing here is submitted. When a candidate wins, copy its parameters into
teamName.py by hand (keep teamName.py self-contained).

Usage
-----
    python loop.py                 # reproduce the current production score
    python loop.py --sweep         # run the built-in parameter search
    python loop.py --sweep --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from dataclasses import dataclass, asdict, field, fields

import numpy as np
import pandas as pd

# ── Official evaluator constants (mirrors eval.py) ────────────────────────────
PRICES_FILE = "./prices.txt"
NUM_TEST_DAYS = 250
SCORE_PARAM = 1.0
DEFAULT_COMM_RATE = 0.0001
INST0_COMM_RATE = 0.00002
DEFAULT_DLR_LIMIT = 10_000
INST0_DLR_LIMIT = 100_000
VOLATILITY_FLOOR = 1e-12


# ── Strategy parameters (the genuine tunables of the production strategy) ─────
@dataclass(frozen=True)
class Params:
    lookbacks: tuple[int, ...] = (5, 20)
    weights: tuple[float, ...] | None = None  # None => equal weight
    rebalance_band: float = 0.195
    algo_dollar_limit: float = 100_000.0
    default_dollar_limit: float = 10_000.0
    signal_clip: float = 1.0
    # H2 momentum overlay (default off — baseline unchanged)
    momentum_lookback: int = 10
    momentum_weight: float = 0.0
    # H7 vol-regime scale (production: cut to 22% when short/long vol ≥ 1.15)
    regime_vol_short: int = 10
    regime_vol_long: int = 60
    regime_threshold: float = 1.15
    regime_scale: float = 0.22
    # Signal EMA blend with prior-day signal (1.0 = off / use today only)
    signal_ema_alpha: float = 1.0
    # Track A: cross-sectional overlay (default off)
    xs_lookback: int = 5
    xs_weight: float = 0.0  # 0 => disabled
    xs_include_algo: bool = False
    # Track B: ALGO-vs-basket residual (simple pairs proxy; production: on)
    pairs_lookback: int = 40
    pairs_weight: float = 0.20  # 0 => disabled
    pairs_entry_z: float = 2.0
    # Track C: ALGO-specific exposure multiplier (1.0 => off; production: 3.0)
    algo_signal_scale: float = 3.0
    # Optional: hedge basket beta with ALGO (0 => off)
    algo_hedge_weight: float = 0.0
    # Track L1: rolling OLS ALGO-vs-basket (default off)
    ols_lookback: int = 40
    ols_weight: float = 0.0
    ols_entry_z: float = 2.0
    ols_intercept: bool = False
    # Track L2: multi-pair OLS (default off)
    mpairs_lookback: int = 60
    mpairs_weight: float = 0.0
    mpairs_top_k: int = 5
    mpairs_entry_z: float = 2.0
    mpairs_min_corr: float = 0.85

    def label(self) -> str:
        w = "eq" if self.weights is None else "/".join(f"{x:.2f}" for x in self.weights)
        mom = (
            f" mom={self.momentum_lookback}@{self.momentum_weight:.2f}"
            if self.momentum_weight
            else ""
        )
        clip = f" clip={self.signal_clip:.2f}" if self.signal_clip != 1.0 else ""
        regime = (
            f" reg={self.regime_vol_short}/{self.regime_vol_long}"
            f"@{self.regime_threshold:.2f}x{self.regime_scale:.2f}"
            if self.regime_scale != 1.0
            else ""
        )
        ema = (
            f" ema={self.signal_ema_alpha:.2f}"
            if self.signal_ema_alpha != 1.0
            else ""
        )
        xs = (
            f" xs={self.xs_lookback}@{self.xs_weight:.2f}"
            f"{'+algo' if self.xs_include_algo else ''}"
            if self.xs_weight > 0
            else ""
        )
        pairs = (
            f" pairs={self.pairs_lookback}@{self.pairs_weight:.2f}"
            f"z{self.pairs_entry_z:.2f}"
            if self.pairs_weight > 0
            else ""
        )
        algo_scale = (
            f" algoscale={self.algo_signal_scale:.2f}"
            if self.algo_signal_scale != 1.0
            else ""
        )
        algo_hedge = (
            f" algohedge={self.algo_hedge_weight:.2f}"
            if self.algo_hedge_weight > 0
            else ""
        )
        ols = (
            f" ols={self.ols_lookback}@{self.ols_weight:.2f}"
            f"z{self.ols_entry_z:.2f}"
            f"{'+int' if self.ols_intercept else ''}"
            if self.ols_weight > 0
            else ""
        )
        mpairs = (
            f" mpairs={self.mpairs_lookback}@{self.mpairs_weight:.2f}"
            f"k{self.mpairs_top_k}z{self.mpairs_entry_z:.2f}"
            f"c{self.mpairs_min_corr:.2f}"
            if self.mpairs_weight > 0
            else ""
        )
        return (
            f"lb={'/'.join(map(str, self.lookbacks))} w={w} "
            f"band={self.rebalance_band:.3f} algo={self.algo_dollar_limit:.0f}"
            f"{clip}{mom}{regime}{ema}{xs}{pairs}{algo_scale}{algo_hedge}"
            f"{ols}{mpairs}"
        )


def rolling_ols_beta(
    y: np.ndarray,
    x: np.ndarray,
    lb: int,
    intercept: bool = False,
) -> float:
    """Least-squares slope of y on x using the last lb observations."""
    yy = np.asarray(y[-lb:], dtype=float)
    xx = np.asarray(x[-lb:], dtype=float)
    if intercept:
        X = np.column_stack([np.ones(lb), xx])
        coef, _, _, _ = np.linalg.lstsq(X, yy, rcond=None)
        return float(coef[1])
    denom = float(xx @ xx)
    if denom < VOLATILITY_FLOOR:
        return 0.0
    return float(xx @ yy) / denom


def spread_z(spread: np.ndarray, lb: int) -> float:
    """Z-score of the last point vs the last lb window."""
    window = np.asarray(spread[-lb:], dtype=float)
    mu = float(window.mean())
    sd = float(window.std()) + VOLATILITY_FLOOR
    return (float(window[-1]) - mu) / sd


# ── Parameterised strategy (a pure function of history + previous positions) ──
def _reversal_signal(prc_so_far: np.ndarray, params: Params) -> np.ndarray:
    """Vol-standardised multi-horizon reversal (+ optional momentum) signal."""
    nins, nt = prc_so_far.shape
    longest = max(params.lookbacks)
    if params.momentum_weight:
        longest = max(longest, params.momentum_lookback)
    if nt <= longest:
        return np.zeros(nins)

    recent = prc_so_far[:, -(longest + 1):]
    log_prices = np.log(recent)
    daily_returns = np.diff(log_prices, axis=1)

    weights = params.weights
    if weights is None:
        weights = tuple(1.0 / len(params.lookbacks) for _ in params.lookbacks)

    signal = np.zeros(nins)
    for lookback, weight in zip(params.lookbacks, weights):
        cum_return = log_prices[:, -1] - log_prices[:, -(lookback + 1)]
        vol = daily_returns[:, -lookback:].std(axis=1) * np.sqrt(lookback)
        standardized = cum_return / np.maximum(vol, VOLATILITY_FLOOR)
        signal -= np.clip(standardized, -params.signal_clip, params.signal_clip) * weight

    if params.momentum_weight:
        mlb = params.momentum_lookback
        mom_ret = log_prices[:, -1] - log_prices[:, -(mlb + 1)]
        mom_vol = daily_returns[:, -mlb:].std(axis=1) * np.sqrt(mlb)
        mom_std = mom_ret / np.maximum(mom_vol, VOLATILITY_FLOOR)
        signal += (
            np.clip(mom_std, -params.signal_clip, params.signal_clip)
            * params.momentum_weight
        )
    return signal


def strategy_positions(
    prc_so_far: np.ndarray,
    prev_positions: np.ndarray,
    params: Params,
) -> np.ndarray:
    """Volatility-standardised multi-horizon reversal with a rebalance band.

    This is the production logic from teamName.py, made parameter-driven so the
    loop can search over it. Returns integer target positions (shares).
    """
    nins, nt = prc_so_far.shape
    longest = max(params.lookbacks)
    if params.momentum_weight:
        longest = max(longest, params.momentum_lookback)
    if params.regime_scale != 1.0:
        longest = max(longest, params.regime_vol_long)
    if params.xs_weight > 0:
        longest = max(longest, params.xs_lookback)
    if params.pairs_weight > 0:
        longest = max(longest, params.pairs_lookback)
    if params.ols_weight > 0:
        longest = max(longest, params.ols_lookback)
    min_days = longest + 1 if params.signal_ema_alpha < 1.0 else longest

    if nt <= min_days:
        return np.zeros(nins, dtype=int)

    signal = _reversal_signal(prc_so_far, params)
    if params.signal_ema_alpha < 1.0:
        prev_signal = _reversal_signal(prc_so_far[:, :-1], params)
        alpha = params.signal_ema_alpha
        signal = alpha * signal + (1.0 - alpha) * prev_signal

    need = max(params.lookbacks)
    if params.regime_scale != 1.0:
        need = max(need, params.regime_vol_long)
    if params.xs_weight > 0:
        need = max(need, params.xs_lookback)
    if params.pairs_weight > 0:
        need = max(need, params.pairs_lookback)
    if params.ols_weight > 0:
        need = max(need, params.ols_lookback)
    recent = prc_so_far[:, -(need + 1):]
    log_prices = np.log(recent)
    daily_returns = np.diff(log_prices, axis=1)

    if params.xs_weight > 0:
        lb = params.xs_lookback
        if nt > lb:
            xs_ret = log_prices[:, -1] - log_prices[:, -(lb + 1)]
            if not params.xs_include_algo:
                xs_ret = xs_ret.copy()
                xs_ret[0] = np.nan
            # Cross-sectional demean (nan-safe)
            mu = np.nanmean(xs_ret)
            xs_signal = -(xs_ret - mu)  # reverse relative winners
            xs_signal = np.nan_to_num(xs_signal, nan=0.0)
            # Standardise cross-sectionally
            sd = np.std(xs_signal) + VOLATILITY_FLOOR
            xs_signal = np.clip(xs_signal / sd, -params.signal_clip, params.signal_clip)
            signal = (1.0 - params.xs_weight) * signal + params.xs_weight * xs_signal

    if params.pairs_weight > 0 and nt > params.pairs_lookback:
        lb = params.pairs_lookback
        # Equal-weight basket of instruments 1..n (ex-ALGO)
        basket = np.nanmean(log_prices[1:, :], axis=0)
        algo = log_prices[0, :]
        spread = algo - basket
        mu_s = spread[-lb:].mean()
        sd_s = spread[-lb:].std() + VOLATILITY_FLOOR
        z = (spread[-1] - mu_s) / sd_s
        if abs(z) >= params.pairs_entry_z:
            pair_sig = np.zeros(nins)
            pair_sig[0] = -np.clip(z, -params.signal_clip, params.signal_clip)
            pair_sig[1:] = -pair_sig[0] / (nins - 1)
            signal = (1.0 - params.pairs_weight) * signal + params.pairs_weight * pair_sig

    if params.ols_weight > 0 and nt > params.ols_lookback:
        lb = params.ols_lookback
        basket = np.nanmean(log_prices[1:, :], axis=0)
        algo = log_prices[0, :]
        beta = rolling_ols_beta(algo, basket, lb, intercept=params.ols_intercept)
        spread = algo - beta * basket
        z = spread_z(spread, lb)
        if abs(z) >= params.ols_entry_z:
            ols_sig = np.zeros(nins)
            ols_sig[0] = -np.clip(z, -params.signal_clip, params.signal_clip)
            ols_sig[1:] = -ols_sig[0] / (nins - 1)
            signal = (1.0 - params.ols_weight) * signal + params.ols_weight * ols_sig

    if params.algo_signal_scale != 1.0:
        signal = signal.copy()
        signal[0] *= params.algo_signal_scale

    if params.algo_hedge_weight > 0:
        basket_sig = float(np.mean(signal[1:]))
        signal = signal.copy()
        signal[0] = signal[0] * (1.0 - params.algo_hedge_weight) - params.algo_hedge_weight * basket_sig

    dollar_limits = np.full(nins, params.default_dollar_limit, dtype=float)
    dollar_limits[0] = params.algo_dollar_limit
    exposure = np.ones(nins)
    if params.regime_scale != 1.0:
        short_n = params.regime_vol_short
        long_n = params.regime_vol_long
        short_vol = daily_returns[:, -short_n:].std(axis=1)
        long_vol = daily_returns[:, -long_n:].std(axis=1)
        ratio = short_vol / np.maximum(long_vol, VOLATILITY_FLOOR)
        exposure = np.where(ratio >= params.regime_threshold, params.regime_scale, 1.0)
    target_dollars = dollar_limits * np.clip(signal, -1.0, 1.0) * exposure
    current_prices = prc_so_far[:, -1]
    desired = np.trunc(target_dollars / current_prices).astype(int)

    rebalance_notional = np.abs(desired - prev_positions) * current_prices
    should_rebalance = rebalance_notional >= params.rebalance_band * dollar_limits
    new_positions = np.where(should_rebalance, desired, prev_positions)

    max_shares = np.trunc(dollar_limits / current_prices).astype(int)
    return np.clip(new_positions, -max_shares, max_shares)


# ── Simulator (faithful reimplementation of eval.py's calcPL/score) ───────────
def load_prices(fn: str = PRICES_FILE) -> np.ndarray:
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    return df.values.T  # (nInst, nDays)


def _score(mu: float, sigma: float, param: float = SCORE_PARAM) -> float:
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac


@dataclass
class Result:
    mean_pl: float
    std_pl: float
    sharpe: float
    score: float
    dvol: float
    pll: np.ndarray = field(repr=False)


def simulate(prc_all: np.ndarray, params: Params, num_test_days: int) -> Result:
    """Day-by-day simulation identical in mechanics to eval.py's calcPL."""
    nins, nt = prc_all.shape

    comm_rate = np.full(nins, DEFAULT_COMM_RATE)
    comm_rate[0] = INST0_COMM_RATE
    dlr_limit = np.full(nins, DEFAULT_DLR_LIMIT)
    dlr_limit[0] = INST0_DLR_LIMIT

    cash = 0.0
    cur_pos = np.zeros(nins)
    tot_dvolume = 0.0
    value = 0.0
    comm = 0.0
    pll: list[float] = []

    start_day = nt - num_test_days
    for t in range(start_day, nt + 1):
        prc_so_far = prc_all[:, :t]
        cur_prices = prc_so_far[:, -1]

        if t < nt:
            new_orig = strategy_positions(prc_so_far, cur_pos.astype(int), params)
            pos_limits = (dlr_limit / cur_prices).astype(int)
            new_pos = np.clip(new_orig, -pos_limits, pos_limits).astype(int)
        else:
            new_pos = np.array(cur_pos)

        delta = new_pos - cur_pos
        cash -= cur_prices.dot(delta) + comm
        dvolumes = cur_prices * np.abs(delta)
        tot_dvolume += np.sum(dvolumes)
        comm = np.sum(dvolumes * comm_rate)

        cur_pos = np.array(new_pos)
        pos_value = cur_pos.dot(cur_prices)
        today_pl = cash + pos_value - value
        value = cash + pos_value

        if t > start_day:
            pll.append(today_pl)

    pll_arr = np.array(pll)
    mu, std = float(np.mean(pll_arr)), float(np.std(pll_arr))
    sharpe = np.sqrt(250) * mu / std if std > 0 else 0.0
    return Result(mu, std, sharpe, _score(mu, std), tot_dvolume, pll_arr)


# ── Robust evaluation: official score + overfit guards ────────────────────────
@dataclass
class Evaluation:
    params: Params
    score: float          # official (last NUM_TEST_DAYS)
    mean_pl: float
    sharpe: float
    dvol: float
    half1_score: float    # first 125 days of the test window
    half2_score: float    # last 125 days of the test window
    train_score: float    # earlier held-out window (not the scored one)
    robust: bool          # passes overfit guards

    def row(self) -> dict:
        d = asdict(self.params)
        d.update(
            label=self.params.label(),
            score=round(self.score, 2),
            mean_pl=round(self.mean_pl, 2),
            sharpe=round(self.sharpe, 3),
            dvol=round(self.dvol, 0),
            half1=round(self.half1_score, 2),
            half2=round(self.half2_score, 2),
            train=round(self.train_score, 2),
            robust=self.robust,
        )
        return d


def evaluate(prc_all: np.ndarray, params: Params) -> Evaluation:
    """Score a candidate on the official window plus overfitting guards."""
    official = simulate(prc_all, params, NUM_TEST_DAYS)

    # Split the official 250-day PnL into two 125-day halves for stability.
    half = len(official.pll) // 2
    h1, h2 = official.pll[:half], official.pll[half:]
    half1 = _score(float(np.mean(h1)), float(np.std(h1)))
    half2 = _score(float(np.mean(h2)), float(np.std(h2)))

    # "Train" window: score an earlier slice the official metric never touches
    # (the 125 days immediately before the official test window), so a candidate
    # cannot win purely by fitting the scored days.
    nt = prc_all.shape[1]
    train_prices = prc_all[:, : nt - NUM_TEST_DAYS]
    train = simulate(train_prices, params, 125)

    robust = (
        official.mean_pl > 0
        and half1 > 0
        and half2 > 0
        and train.mean_pl > 0
    )
    return Evaluation(
        params=params,
        score=official.score,
        mean_pl=official.mean_pl,
        sharpe=official.sharpe,
        dvol=official.dvol,
        half1_score=half1,
        half2_score=half2,
        train_score=train.score,
        robust=robust,
    )


# ── The loop: sweep, log, leaderboard, promote ────────────────────────────────
def build_grid() -> list[Params]:
    """Track L1: rolling OLS ALGO-basket. Prefer replace mode (pairs_weight=0)."""
    grid = [Params()]  # production floor
    # Replace mode: turn off β≈1 pairs, put weight on ols_*
    for lb in (30, 40, 60):
        for w in (0.10, 0.20, 0.30):
            for z in (1.5, 2.0, 2.5):
                grid.append(Params(
                    pairs_weight=0.0,
                    ols_lookback=lb,
                    ols_weight=w,
                    ols_entry_z=z,
                    ols_intercept=False,
                ))
    # Additive smoke: keep production pairs, add one OLS candidate
    grid.append(Params(
        ols_lookback=40, ols_weight=0.10, ols_entry_z=2.0, ols_intercept=False,
    ))
    # Intercept variant (replace)
    grid.append(Params(
        pairs_weight=0.0,
        ols_lookback=40, ols_weight=0.20, ols_entry_z=2.0, ols_intercept=True,
    ))
    return grid


def _result_fieldnames() -> list[str]:
    """Stable CSV column order: all Params fields, then evaluation metrics."""
    param_keys = [f.name for f in fields(Params)]
    eval_keys = ("label", "score", "mean_pl", "sharpe", "dvol", "half1", "half2", "train", "robust")
    return param_keys + list(eval_keys)


def log_results(rows: list[dict], path: str) -> None:
    if not rows:
        return
    fieldnames = _result_fieldnames()
    exists = os.path.exists(path)
    header_ok = False
    if exists:
        with open(path, newline="") as f:
            header_ok = next(csv.reader(f), None) == fieldnames
    mode = "a" if exists and header_ok else "w"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def is_promotable(base: Evaluation, cand: Evaluation) -> bool:
    """A candidate is promotable only if it beats the baseline official score,
    passes every robustness guard, and does not sacrifice the weaker test half.
    This is the single source of truth for the agent's promote/reject decision.
    """
    return (
        cand.robust
        and cand.score > base.score
        and cand.half2_score >= base.half2_score * 0.95
    )


def run_sweep(prc_all: np.ndarray, baseline: Params, csv_path: str | None) -> None:
    base_eval = evaluate(prc_all, baseline)
    print(f"\nBaseline: {baseline.label()}")
    print(
        f"  score={base_eval.score:.2f}  sharpe={base_eval.sharpe:.3f}  "
        f"half1={base_eval.half1_score:.2f}  half2={base_eval.half2_score:.2f}  "
        f"train={base_eval.train_score:.2f}\n"
    )

    grid = build_grid()
    evals = [evaluate(prc_all, p) for p in grid]
    evals.sort(key=lambda e: e.score, reverse=True)

    print(f"{'score':>7} {'sharpe':>7} {'half1':>7} {'half2':>7} {'train':>7} {'ok':>3}  params")
    print("-" * 80)
    for e in evals:
        flag = "Y" if e.robust else "-"
        print(
            f"{e.score:7.2f} {e.sharpe:7.3f} {e.half1_score:7.2f} "
            f"{e.half2_score:7.2f} {e.train_score:7.2f} {flag:>3}  {e.params.label()}"
        )

    if csv_path:
        log_results([e.row() for e in evals], csv_path)
        print(f"\nLogged {len(evals)} candidates to {csv_path}")

    winners = [e for e in evals if is_promotable(base_eval, e)]
    print("\n" + "=" * 80)
    if winners:
        best = winners[0]
        gain = best.score - base_eval.score
        print(f"PROMOTE  {best.params.label()}")
        print(f"  official score {base_eval.score:.2f} -> {best.score:.2f}  (+{gain:.2f})")
        print("  -> copy these params into teamName.py, then run: python eval.py")
    else:
        print("NO PROMOTION  baseline remains best robust candidate.")
        print("  Add new hypotheses to build_grid() and re-run.")
    print("=" * 80)


def verdict_json(prc_all: np.ndarray, baseline: Params, grid: list[Params]) -> dict:
    """Machine-readable verdict for an AI agent to parse and act on.

    Returns the baseline, every scored candidate, and whether the best robust
    candidate should be promoted. `--json` prints exactly this, so an agent's
    promote/reject decision is deterministic instead of eyeballed.
    """
    base_eval = evaluate(prc_all, baseline)
    evals = sorted(
        (evaluate(prc_all, p) for p in grid), key=lambda e: e.score, reverse=True
    )
    winners = [e for e in evals if is_promotable(base_eval, e)]
    best = winners[0] if winners else None
    return {
        "baseline": base_eval.row(),
        "candidates": [e.row() for e in evals],
        "promote": best is not None,
        "winner": best.row() if best else None,
        "gain": round(best.score - base_eval.score, 2) if best else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Algothon strategy iteration loop")
    parser.add_argument("--sweep", action="store_true", help="run the parameter search")
    parser.add_argument("--json", action="store_true", help="emit machine-readable verdict (for AI agents)")
    parser.add_argument("--csv", default="results.csv", help="results log path")
    args = parser.parse_args()

    prc_all = load_prices()
    baseline = Params()  # matches current teamName.py

    if args.json:
        print(json.dumps(verdict_json(prc_all, baseline, build_grid()), indent=2))
        return

    print(f"Loaded {prc_all.shape[0]} instruments for {prc_all.shape[1]} days")
    if args.sweep:
        run_sweep(prc_all, baseline, args.csv)
    else:
        e = evaluate(prc_all, baseline)
        print(f"\nProduction strategy: {baseline.label()}")
        print(f"  mean(PL): {e.mean_pl:.1f}")
        print(f"  annSharpe: {e.sharpe:.2f}")
        print(f"  Score: {e.score:.2f}")
        print(f"  half1/half2/train: {e.half1_score:.2f} / {e.half2_score:.2f} / {e.train_score:.2f}")
        print("\nRun `python loop.py --sweep` to search for improvements.")


if __name__ == "__main__":
    main()
