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
