from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loop import Params, load_prices, simulate_range


def warmup_days(p: Params) -> int:
    longest = max(max(p.lookbacks), p.regime_vol_long, p.mpairs_lookback, p.ols_lookback)
    if p.adaptive_band_vol_lb > 0:
        longest = max(longest, p.adaptive_band_vol_lb)
    if p.vol_target_lookback > 0:
        longest = max(longest, p.vol_target_lookback)
    if p.leadlag_weight > 0:
        lag_need = p.leadlag_min_obs + 1
        if p.leadlag_lookback > 0:
            lag_need = max(lag_need, p.leadlag_lookback + 1)
        longest = max(longest, lag_need)
    if p.edgepairs_weight > 0 and p.edgepairs_select_lookback > 0:
        longest = max(
            longest,
            p.edgepairs_select_lookback + 1,
            p.edgepairs_fit_lookback,
        )
    if p.factor_weight > 0 and p.factor_lookback > 0:
        longest = max(longest, p.factor_lookback)
    return longest


def rolling_starts(nt: int, p: Params, step: int = 25) -> list[int]:
    last_start = nt - 250
    first_start = min(max(warmup_days(p), 1), last_start)
    starts = list(range(first_start, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def score_candidate(prc: np.ndarray, name: str, p: Params) -> dict[str, object]:
    starts = rolling_starts(prc.shape[1], p)
    results = [simulate_range(prc, p, s, s + 250) for s in starts]
    scores = np.array([r.score for r in results], dtype=float)
    means = np.array([r.mean_pl for r in results], dtype=float)
    sharpes = np.array([r.sharpe for r in results], dtype=float)
    official = results[-1]
    return {
        "name": name,
        "params": p,
        "official": official.score,
        "mean_pl": official.mean_pl,
        "std_pl": official.std_pl,
        "sharpe": official.sharpe,
        "min": float(scores.min()),
        "median": float(np.median(scores)),
        "avg": float(scores.mean()),
        "early": float(scores[0]),
        "recent100": simulate_range(prc, p, prc.shape[1] - 100, prc.shape[1]).score,
        "scores": tuple(float(x) for x in scores),
        "means": tuple(float(x) for x in means),
        "sharpes": tuple(float(x) for x in sharpes),
    }


def print_rows(rows: list[dict[str, object]]) -> None:
    rows = sorted(rows, key=lambda r: (r["min"], r["avg"], r["official"]), reverse=True)
    print(
        f"{'name':<34} {'off':>8} {'min':>8} {'med':>8} "
        f"{'avg':>8} {'early':>8} {'r100':>8} {'sr':>6}"
    )
    print("-" * 98)
    for r in rows:
        print(
            f"{str(r['name'])[:34]:<34} "
            f"{r['official']:8.2f} {r['min']:8.2f} {r['median']:8.2f} "
            f"{r['avg']:8.2f} {r['early']:8.2f} {r['recent100']:8.2f} "
            f"{r['sharpe']:6.2f}"
        )
        print("  " + " / ".join(f"{x:.0f}" for x in r["scores"]))


def main() -> None:
    prc = load_prices()
    base = Params()
    candidates: list[tuple[str, Params]] = [("base", base)]
    for power in (1.30, 1.35, 1.40, 1.45, 1.50):
        for shrink in (0.015, 0.018, 0.020, 0.022, 0.025):
            candidates.append(
                (
                    f"p{power:.2f} s{shrink:.3f}",
                    replace(base, leadlag_power=power, leadlag_shrink=shrink),
                )
            )

    rows = []
    for idx, (name, p) in enumerate(candidates, start=1):
        row = score_candidate(prc, name, p)
        rows.append(row)
        print(
            f"{idx:02d}/{len(candidates)} {name:<28} "
            f"off={row['official']:.2f} min={row['min']:.2f} avg={row['avg']:.2f}",
            flush=True,
        )
    print()
    print_rows(rows[:])


if __name__ == "__main__":
    main()
