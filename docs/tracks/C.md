# Track C — ALGO-centric / hedge

**Status:** active  
**Owner:** Agent C  
**Params prefix:** `algo_*` extras beyond `algo_dollar_limit` (default off / production values)  
**Kill rule:** 3 consecutive non-promotable ticks → freeze

## Hypothesis bank
1. Stronger standalone signal on instrument 0 only; others keep production.
2. Use ALGO to hedge basket beta of the reversal book.
3. Asymmetric ALGO sizing vs others (already know cutting cap to $50K hurt — do not repeat).

## Reject memory
- Halving ALGO cap → score 133.98. Full $100K is productive.

## Log
| Tick | Hypothesis | Best score | train | half2 | Verdict |
|:---|:---|---:|---:|---:|:---|
| 1 | Added `algo_signal_scale` / `algo_hedge_weight` Params (default off) applied to the ALGO leg of the signal after xs/pairs overlays, before dollar sizing; screened scale ∈ {0.5,0.75,1.25,1.5,2.0} and hedge weight ∈ {0.25,0.50,0.75,1.0}. Baseline (scale=1.0, hedge=0.0) reproduces 211.49 exactly. | 228.65 (algoscale=2.00) | 15.77 | 155.53 | PROMOTABLE per `is_promotable` (score 211.49→228.65, half2 155.53≥139.4) — **held back**, not copied to teamName.py per track isolation instructions; flag for coordinator. |

### Tick 1 detail
All 18 grid candidates (incl. baseline) passed robustness guards (mean_pl>0, half1>0, half2>0, train>0):

| algo_signal_scale | algo_hedge_weight | score | half1 | half2 | train |
|---:|---:|---:|---:|---:|---:|
| 2.00 | — | 228.65 | 301.45 | 155.53 | 15.77 |
| 1.50 | — | 224.90 | 303.66 | 145.68 | 9.93 |
| 1.25 | — | 214.07 | 289.85 | 137.82 | 12.89 |
| 1.00 (baseline) | 0.00 | 211.49 | 275.99 | 146.75 | 10.19 |
| 0.75 | — | 187.20 | 242.75 | 131.53 | 16.03 |
| — | 0.25 | 179.56 | 235.35 | 123.70 | 17.01 |
| 0.50 | — | 176.97 | 223.47 | 130.50 | 20.83 |
| — | 0.50 | 162.45 | 198.03 | 127.09 | 22.33 |
| — | 0.75 | 132.57 | 165.41 | 100.02 | 34.23 |
| — | 1.00 | 130.55 | 136.72 | 124.61 | 34.63 |

`algo_signal_scale` monotonically improves score across the entire tested range (0.5→228.65 as scale rises to 2.0) — i.e. simply amplifying the existing ALGO signal before the `[-1,1]` dollar-sizing clip makes ALGO hit its $100K cap more often, and that's net-positive because ALGO already carries a robust edge (see Track C hypothesis 3 / Iteration 5 reject-memory: cutting ALGO's cap hurt, so raising its effective conviction helps in the same direction). **This is a grid-boundary result, not a settled local optimum** — scale=2.0 is the edge of the screened range, so the true optimum (or a diminishing-returns point) is unconfirmed; a future tick should extend the grid (e.g. scale ∈ {2.5, 3.0, 4.0}) to find where returns flatten or half2/train start degrading, since unbounded scale-up eventually must saturate against the ALGO dollar cap and could reintroduce concentration risk. `algo_hedge_weight` is monotonically bad — blending in the negative basket-signal residual only dilutes ALGO's own (already-good) time-series reversal edge; discard this arm.

`loop.py --sweep`/`--json` both confirm `promote: true` (+17.15) and 8/8 existing tests pass with the new default-off fields. **Per task instructions this track does not promote to `teamName.py`** — logged here and in `STRATEGY_LOOP.md` for the coordinating process to reconcile against Track A/B/D results before any production change.
