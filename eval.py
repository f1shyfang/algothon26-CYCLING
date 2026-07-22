#!/usr/bin/env python3
"""Local Algothon evaluator with warm-up-aware validation.

The official local score is the last 250 scored days of prices.txt. The extra
windows below are diagnostics only: the live leaderboard scores hidden future
days, so no local slice can estimate it exactly.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


PRICES_FILE = "./prices.txt"
TEAM_MODULE = "teamName"
NUM_TEST_DAYS = 250
MIN_DOLLAR_VOLUME = 25_000

SCORE_DEFAULT_PARAM = 1.0
DEFAULT_COMM_RATE = 0.0001
INST0_COMM_RATE = 0.00002
DEFAULT_DLR_POS_LIMIT = 10_000
INST0_DLR_POS_LIMIT = 100_000


@dataclass
class Result:
    start_day: int
    end_day: int
    mean_pl: float
    std_pl: float
    sharpe: float
    dvol: float
    raw_score: float
    score: float
    inactive: bool
    pll: np.ndarray


def load_prices(fn: str) -> np.ndarray:
    """Load prices as one instrument per row."""
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    return df.values.T


def score(mu: float, sigma: float, param: float = SCORE_DEFAULT_PARAM) -> float:
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac


def fresh_get_position(module_name: str):
    """Reload the strategy so independent validation windows start clean."""
    if module_name in sys.modules:
        module = importlib.reload(sys.modules[module_name])
    else:
        module = importlib.import_module(module_name)
    return module, module.getMyPosition


def infer_warmup_days(module_name: str) -> int:
    """Infer this repo's longest history requirement from strategy constants."""
    module, _ = fresh_get_position(module_name)

    def const(name: str, default):
        return getattr(module, name, default)

    lookbacks = tuple(const("LOOKBACKS", (0,)))
    longest = max(lookbacks) if lookbacks else 0
    longest = max(
        longest,
        int(const("REGIME_VOL_LONG", 0)),
        int(const("MPAIRS_LOOKBACK", 0)),
        int(const("OLS_LOOKBACK", 0)),
    )

    adaptive_lb = int(const("ADAPTIVE_BAND_VOL_LB", 0))
    if adaptive_lb > 0:
        longest = max(longest, adaptive_lb)

    vol_target_lb = int(const("VOL_TARGET_LOOKBACK", 0))
    if vol_target_lb > 0:
        longest = max(longest, vol_target_lb)

    if float(const("LEADLAG_WEIGHT", 0.0)) > 0:
        lag_need = int(const("LEADLAG_MIN_OBS", 0)) + 1
        leadlag_lb = int(const("LEADLAG_LOOKBACK", 0))
        if leadlag_lb > 0:
            lag_need = max(lag_need, leadlag_lb + 1)
        longest = max(longest, lag_need)

    if (
        float(const("EDGEPAIRS_WEIGHT", 0.0)) > 0
        and int(const("EDGEPAIRS_SELECT_LOOKBACK", 0)) > 0
    ):
        longest = max(
            longest,
            int(const("EDGEPAIRS_SELECT_LOOKBACK", 0)) + 1,
            int(const("EDGEPAIRS_FIT_LOOKBACK", 0)),
        )

    return longest


def calc_pl(
    prc_hist: np.ndarray,
    start_day: int,
    end_day: int,
    module_name: str,
    verbose: bool = False,
) -> Result:
    """Calculate PnL from start_day..end_day and score t > start_day.

    The day values are history lengths, matching the official eval mechanics:
    prcSoFar passed to getMyPosition is prc_hist[:, :t].
    """
    ninst = prc_hist.shape[0]
    _, get_position = fresh_get_position(module_name)

    comm_rate = np.full(ninst, DEFAULT_COMM_RATE)
    comm_rate[0] = INST0_COMM_RATE
    dlr_pos_limit = np.full(ninst, DEFAULT_DLR_POS_LIMIT)
    dlr_pos_limit[0] = INST0_DLR_POS_LIMIT

    cash = 0.0
    cur_pos = np.zeros(ninst)
    tot_dvolume = 0.0
    value = 0.0
    comm = 0.0
    pll: list[float] = []

    for t in range(start_day, end_day + 1):
        prc_so_far = prc_hist[:, :t]
        cur_prices = prc_so_far[:, -1]

        if t < end_day:
            new_pos_orig = np.asarray(get_position(prc_so_far))
            if new_pos_orig.shape != (ninst,):
                raise ValueError(
                    f"{module_name}.getMyPosition returned shape "
                    f"{new_pos_orig.shape}, expected {(ninst,)}"
                )
            pos_limits = (dlr_pos_limit / cur_prices).astype(int)
            new_pos = np.clip(new_pos_orig, -pos_limits, pos_limits).astype(int)
        else:
            new_pos = np.array(cur_pos)

        delta_pos = new_pos - cur_pos
        cash -= cur_prices.dot(delta_pos) + comm

        dvolumes = cur_prices * np.abs(delta_pos)
        tot_dvolume += float(np.sum(dvolumes))
        comm = float(np.sum(dvolumes * comm_rate))

        cur_pos = np.array(new_pos)
        pos_value = float(cur_pos.dot(cur_prices))
        today_pl = cash + pos_value - value
        value = cash + pos_value

        if t > start_day:
            if verbose:
                ret = value / tot_dvolume if tot_dvolume > 0 else 0.0
                print(
                    f"Day {t:4d} value: {value:10.2f} "
                    f"todayPL: {today_pl:9.2f} "
                    f"$-traded: {tot_dvolume:11.0f} return: {ret:.5f}"
                )
            pll.append(today_pl)

    pll_arr = np.array(pll)
    mean_pl = float(np.mean(pll_arr))
    std_pl = float(np.std(pll_arr))
    sharpe = np.sqrt(250) * mean_pl / std_pl if std_pl > 0 else 0.0
    raw_score = float(score(mean_pl, std_pl))
    inactive = tot_dvolume < MIN_DOLLAR_VOLUME
    final_score = 0.0 if inactive else raw_score

    return Result(
        start_day=start_day,
        end_day=end_day,
        mean_pl=mean_pl,
        std_pl=std_pl,
        sharpe=sharpe,
        dvol=tot_dvolume,
        raw_score=raw_score,
        score=final_score,
        inactive=inactive,
        pll=pll_arr,
    )


def validation_starts(
    nt: int,
    warmup_days: int,
    num_test_days: int,
    step: int,
) -> list[int]:
    last_start = nt - num_test_days
    if last_start < 1:
        return []

    first_start = min(max(warmup_days, 1), last_start)
    starts = list(range(first_start, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def recent_folds(nt: int, fold_days: int, count: int) -> list[tuple[int, int]]:
    first = nt - fold_days * count
    folds = []
    for i in range(count):
        start = first + i * fold_days
        end = start + fold_days
        if start >= 1 and end <= nt:
            folds.append((start, end))
    return folds


def print_result(label: str, result: Result) -> None:
    flag = " inactive" if result.inactive else ""
    print(
        f"{label:<18} {result.start_day:4d}-{result.end_day:<4d} "
        f"{len(result.pll):4d} "
        f"{result.score:9.2f} {result.mean_pl:10.1f} "
        f"{result.std_pl:10.2f} {result.sharpe:8.2f} "
        f"{result.dvol:13.0f}{flag}"
    )


def print_table_header() -> None:
    print(
        f"{'window':<18} {'days':<9} {'nPL':>4} "
        f"{'score':>9} {'meanPL':>10} {'stdPL':>10} "
        f"{'sharpe':>8} {'dVolume':>13}"
    )
    print("-" * 91)


def summarize(name: str, results: list[Result]) -> None:
    if not results:
        return
    scores = np.array([r.score for r in results], dtype=float)
    print(
        f"{name}: min={scores.min():.2f}, median={np.median(scores):.2f}, "
        f"mean={scores.mean():.2f}, max={scores.max():.2f}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run local Algothon evaluation")
    parser.add_argument("--prices", default=PRICES_FILE, help="price file path")
    parser.add_argument("--module", default=TEAM_MODULE, help="strategy module")
    parser.add_argument("--verbose", action="store_true", help="print daily official PnL")
    parser.add_argument(
        "--window-step",
        type=int,
        default=25,
        help="spacing between warm-up-aware 250-day validation starts",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="only run the official local last-250 score",
    )
    args = parser.parse_args(argv)

    prc_all = load_prices(args.prices)
    ninst, nt = prc_all.shape
    print(f"Loaded {ninst} instruments for {nt} days")

    official_start = nt - NUM_TEST_DAYS
    official = calc_pl(
        prc_all,
        official_start,
        nt,
        module_name=args.module,
        verbose=args.verbose,
    )

    print("\nOFFICIAL LOCAL SCORE")
    print_table_header()
    print_result("last 250", official)

    if args.skip_validation:
        return

    warmup_days = infer_warmup_days(args.module)
    print(
        f"\nInferred warm-up: {warmup_days} days. "
        "Validation windows start at or after this point."
    )

    starts = validation_starts(nt, warmup_days, NUM_TEST_DAYS, args.window_step)
    validation_results = [
        calc_pl(prc_all, start, start + NUM_TEST_DAYS, module_name=args.module)
        for start in starts
    ]

    print("\nWARM-UP-AWARE 250-DAY WINDOWS")
    print_table_header()
    for result in validation_results:
        label = "official" if result.start_day == official_start else "validation"
        print_result(label, result)
    summarize("250-day validation scores", validation_results)

    folds = recent_folds(nt, fold_days=100, count=3)
    fold_results = [
        calc_pl(prc_all, start, end, module_name=args.module)
        for start, end in folds
        if start >= warmup_days
    ]

    if fold_results:
        print("\nRECENT 100-DAY FOLDS")
        print_table_header()
        for idx, result in enumerate(fold_results, start=1):
            print_result(f"fold {idx}", result)
        summarize("100-day fold scores", fold_results)

    print(
        "\nNote: the live leaderboard uses hidden future days. These windows are "
        "stress tests, not a website-score estimate."
    )


if __name__ == "__main__":
    main()
