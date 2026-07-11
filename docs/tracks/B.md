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
