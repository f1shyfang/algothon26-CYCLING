# Track A — Cross-sectional mean reversion

**Status:** active (Day 1 tick 1 complete, 1 consecutive reject — 2 more before kill)  
**Owner:** Agent A  
**Params prefix:** `xs_*` (default off → production unchanged)  
**Kill rule:** 3 consecutive non-promotable ticks → freeze  
**Next (Day 2):** try a rank-based xs formulation (not demean+std); freeze if no genuinely new idea survives scrutiny. See `docs/tracks/PROTOCOL.md`.

## Hypothesis bank
1. Rank instruments by 5d return; buy losers / sell winners (exclude ALGO or include).
2. Vol-standardised cross-sectional z-score over 5d/10d.
3. Blend 80% time-series + 20% cross-sectional (prior 80/20 failed — only retry with different lookback/weight).

## Reject memory
- Prior ensemble H3 overlay (80/20, 5d rank) scored 127.22 — weak standalone, corr 0.49. Do not repeat exact setup.

## Log
| Tick | Hypothesis | Best score | train | half2 | Verdict |
|:---|:---|---:|---:|---:|:---|
| 1 | Added `xs_*` Params (default off) + xs overlay in `strategy_positions`; screened blend of 80–90% TS signal with 10–20% cross-sectional 5d/10d reversal signal, with/without ALGO included. Baseline (xs_weight=0.0) reproduces 211.49 exactly. | 185.00 (xs=10@0.20+algo) | 7.14 | 147.11 | REJECT — every xs blend variant scores below baseline (154.64–185.00 vs 211.49); no promotion. |

### Tick 1 detail
All 8 grid candidates passed robustness guards (mean_pl>0, half1>0, half2>0, train>0) but none beat the baseline official score of 211.49:

| xs_lookback | xs_weight | include_algo | score | half1 | half2 | train |
|---:|---:|:---:|---:|---:|---:|---:|
| 10 | 0.20 | yes | 185.00 | 223.22 | 147.11 | 7.14 |
| 10 | 0.20 | no | 184.50 | 221.10 | 148.23 | 5.52 |
| 10 | 0.10 | yes | 179.77 | 232.92 | 126.74 | 8.03 |
| 10 | 0.10 | no | 177.07 | 232.60 | 121.63 | 7.75 |
| 5 | 0.10 | yes | 173.94 | 226.08 | 121.86 | 12.92 |
| 5 | 0.10 | no | 171.62 | 226.42 | 116.87 | 12.54 |
| 5 | 0.20 | no | 159.65 | 203.36 | 116.05 | 13.08 |
| 5 | 0.20 | yes | 154.64 | 193.88 | 115.58 | 15.44 |

Simple demeaned-reversal cross-sectional overlay consistently drags the official score down (~12–27%) relative to the pure time-series baseline, even at low blend weights (10%). The direction (worse with higher xs_weight, better with 10d lookback vs 5d) suggests the cross-sectional signal is largely redundant with / noisier than the existing per-instrument vol-standardised reversal, rather than adding orthogonal alpha. Next hypothesis should try a genuinely different xs formulation (e.g. rank-based rather than demean+std, or a much smaller blend weight ≤0.05) rather than retrying this exact family.
