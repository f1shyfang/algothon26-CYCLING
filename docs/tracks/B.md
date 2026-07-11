# Track B — Pairs / cointegration

**Status:** active  
**Owner:** Agent B  
**Params prefix:** `pairs_*` (default off)  
**Kill rule:** 3 consecutive non-promotable ticks → freeze

## Hypothesis bank
1. Top correlated pairs (ρ>0.7 on train window) trade spread z-score.
2. ALGO vs equal-weight basket residual mean reversion.
3. Small number of pairs (≤5) to limit turnover.

## Reject memory
(none yet)

## Log
| Tick | Hypothesis | Best score | train | half2 | Verdict |
|:---|:---|---:|---:|---:|:---|
| 1 | Added `pairs_*` Params (default off) + ALGO-vs-equal-weight-basket residual overlay (log-price spread z-score, mean-reversion) in `strategy_positions`; screened lookback {20,40} × weight {0.10,0.20} × entry-z {1.0,1.5,2.0}. Baseline (pairs_weight=0.0) reproduces 211.49 exactly. | 219.30 (pairs=40@0.20z2.00) | 11.11 | 148.12 | PROMOTABLE per `is_promotable` (score 211.49→219.30, half2 148.12≥139.4) — **held back**, not copied to teamName.py per track isolation instructions; flag for coordinator. |

### Tick 1 detail
All 9 grid candidates (incl. baseline) passed robustness guards (mean_pl>0, half1>0, half2>0, train>0):

| pairs_lookback | pairs_weight | pairs_entry_z | score | half1 | half2 | train |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.20 | 2.0 | 219.30 | 289.97 | 148.12 | 11.11 |
| 20 | 0.20 | 2.0 | 216.57 | 268.62 | 164.56 | 10.97 |
| 20 (baseline, w=0) | — | — | 211.49 | 275.99 | 146.75 | 10.19 |
| 20 | 0.10 | 1.5 | 201.92 | 262.14 | 141.45 | 8.76 |
| 20 | 0.20 | 1.5 | 199.19 | 269.45 | 128.38 | 8.56 |
| 40 | 0.10 | 1.5 | 193.70 | 251.77 | 135.46 | 6.50 |
| 40 | 0.10 | 1.0 | 179.77 | 243.92 | 115.50 | 7.55 |
| 20 | 0.10 | 1.0 | 177.96 | 231.62 | 124.30 | 7.08 |
| 40 | 0.20 | 1.5 | 174.61 | 231.98 | 116.94 | 2.39 |

The higher-entry-z, higher-weight variants (z=2.0, w=0.20) win: a wide spread threshold means the overlay only fires on genuinely stretched ALGO-vs-basket divergences, and a large blend weight lets those rare signals dominate on the days they fire. Tighter thresholds (z=1.0–1.5) fire too often on noise and drag the score below baseline. `loop.py --sweep`/`--json` both confirm `promote: true` (+7.81) and 8/8 existing tests pass with the new default-off fields. **Per task instructions this track does not promote to `teamName.py`** — logged here and in `STRATEGY_LOOP.md` for the coordinating process to reconcile against Track A/C/D results before any production change.
