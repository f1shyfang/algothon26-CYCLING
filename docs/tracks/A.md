# Track A — Cross-sectional mean reversion

**Status:** active  
**Owner:** Agent A  
**Params prefix:** `xs_*` (default off → production unchanged)  
**Kill rule:** 3 consecutive non-promotable ticks → freeze

## Hypothesis bank
1. Rank instruments by 5d return; buy losers / sell winners (exclude ALGO or include).
2. Vol-standardised cross-sectional z-score over 5d/10d.
3. Blend 80% time-series + 20% cross-sectional (prior 80/20 failed — only retry with different lookback/weight).

## Reject memory
- Prior ensemble H3 overlay (80/20, 5d rank) scored 127.22 — weak standalone, corr 0.49. Do not repeat exact setup.

## Log
| Tick | Hypothesis | Best score | train | half2 | Verdict |
|:---|:---|---:|---:|---:|:---|
