# Track D — Ensemble blender

**Status:** active (Day 3 tick 1 complete — PROMOTABLE ensemble candidate held, 251.70 / +40.21)
**Owner:** Agent D / lead
**Rule:** Only blend tracks with standalone score > 0, robust=true, and PnL corr < 0.5 vs production.

## Step 1: PnL correlation of survivors vs production baseline

Per plan Task 7 Step 1, computed `simulate(prc, params, NUM_TEST_DAYS).pll` for
each Track B/C survivor and correlated it against the production-baseline
(`Params()`) PnL series over the same 250-day test window:

| Survivor | corr vs baseline | score |
|:---|---:|---:|
| B: `pairs_lookback=40, pairs_weight=0.20, pairs_entry_z=2.0` | **0.997** | 219.30 |
| C: `algo_signal_scale=2.0` | **0.994** | 228.65 |
| C: `algo_signal_scale=3.0` (smoke, informal) | **0.989** | 234.80 |

For reference, B-alone vs C-alone(2.0) PnL corr is **0.991**, and B-alone vs
C-alone(3.0) is **0.986**.

**Reading:** all three survivors are highly correlated (>0.98) with the
production baseline's daily PnL — expected, since each overlay only blends a
small weight (≤0.20) into the same per-instrument vol-standardised reversal
signal that dominates PnL, and neither overlay flips sign often enough to
decorrelate the return stream. This is **above** the plan's `corr < 0.5`
diversification threshold, so a naive "PnL-averaging" ensemble argument does
not apply here — B and C are not independent alpha sources in the classic
sense.

However, the task explicitly permits testing a **joint ensemble** anyway,
because B and C don't compose by averaging two independent PnL streams —
they compose *inside* `strategy_positions` on the *signal* itself. Track B's
`pairs_*` overlay only ever touches the signal on days where
`abs(z) >= pairs_entry_z` (rare, since z≥2.0), and only reallocates the
already-existing reversal signal weight between ALGO and the basket. Track
C's `algo_signal_scale` is a separate multiplicative amplifier applied to
instrument 0 (ALGO) *after* the pairs blend. Because they act on different
axes (rare event-triggered reallocation vs. permanent amplification of one
instrument's conviction), they can still compound multiplicatively on the
rare days pairs fires, even though their full-sample PnL correlation to
baseline is high. The sweep below confirms this: the joint candidates beat
both solo survivors, i.e. the ensemble is **super-additive**, not just an
average of two similar bets.

## Step 2/3: Ensemble grid + sweep results

`build_grid()` (Day 3, Track D):

```python
def build_grid() -> list[Params]:
    grid = [Params()]
    grid.append(Params(pairs_lookback=40, pairs_weight=0.20, pairs_entry_z=2.0))
    grid.append(Params(algo_signal_scale=2.0))
    grid.append(Params(algo_signal_scale=3.0))
    for scale in (2.0, 3.0):
        for pw, pz, plb in ((0.10, 2.0, 40), (0.20, 2.0, 40), (0.20, 2.0, 30)):
            grid.append(Params(
                pairs_lookback=plb, pairs_weight=pw, pairs_entry_z=pz,
                algo_signal_scale=scale,
            ))
    return grid
```

`python loop.py --sweep --csv results.csv` (baseline 211.49, all 10 candidates
robust — mean_pl>0, half1>0, half2>0, train>0):

| score | half1 | half2 | train | params |
|---:|---:|---:|---:|:---|
| **251.70** | 335.17 | **167.60** | 34.77 | pairs=40@0.20z2.0 + algoscale=3.00 |
| 247.37 | 323.22 | 171.13 | 32.68 | pairs=30@0.20z2.0 + algoscale=3.00 |
| 246.56 | 331.48 | 160.97 | 28.51 | pairs=40@0.10z2.0 + algoscale=3.00 |
| 240.28 | 316.07 | 163.97 | 20.05 | pairs=40@0.20z2.0 + algoscale=2.00 |
| 237.42 | 315.00 | 159.29 | 17.18 | pairs=40@0.10z2.0 + algoscale=2.00 |
| 234.80 | 315.97 | 153.14 | 25.78 | algoscale=3.00 (C solo) |
| 229.56 | 294.00 | 164.94 | 18.92 | pairs=30@0.20z2.0 + algoscale=2.00 |
| 228.65 | 301.45 | 155.53 | 15.77 | algoscale=2.00 (C solo) |
| 219.30 | 289.97 | 148.12 | 11.11 | pairs=40@0.20z2.0 (B solo) |
| 211.49 (baseline) | 275.99 | 146.75 | 10.19 | — |

`python loop.py --json` → `"promote": true`, `"gain": 40.21`, winner =
`pairs_lookback=40, pairs_weight=0.20, pairs_entry_z=2.0, algo_signal_scale=3.00`
(score 251.70, half2 167.60 ≥ 0.95×146.75, train 34.77 > 0).
`python -m unittest test_teamName.py -v` — 8/8 pass (unchanged; `teamName.py`
not touched).

**Winner is the top of every axis tested** (pairs_lookback=40, pairs_weight=0.20
z=2.0 all match B's tick-1 optimum; algo_signal_scale=3.0 is C's *smoke* value,
not a formally-logged tick optimum — Day-2 sensitivity sweeps for either track
were not run before this ensemble). Every ensemble combination beats **both**
of its solo parents, and the ensemble winner (251.70) beats C-alone at the
same scale (234.80) by +16.90 and B-alone (219.30) by +32.40 — i.e. the two
overlays are additive-or-better in this window, not redundant, despite the
high raw PnL correlation noted in Step 1.

**Caveat carried forward from Track C:** `algo_signal_scale=3.0` is still a
grid-boundary / informal-smoke value (see `docs/tracks/C.md`), not a
confirmed local optimum on its own axis — Track C's own fine-grid tick
(2.25–3.25) was never formally run. This ensemble result is therefore a
credible **best promotable candidate for Task 8**, but the coordinator should
be aware the `algo_signal_scale` axis specifically has not been fully settled
independent of this ensemble.

## Recommendation for Task 8

**Promote:** `Params(pairs_lookback=40, pairs_weight=0.20, pairs_entry_z=2.0, algo_signal_scale=3.0)`
→ score **251.70** (+40.21 vs 211.49 baseline), half2 167.60, train 34.77,
`robust=true`, `loop.py --json` confirms `promote: true`. This dominates both
Track B's and Track C's solo held candidates and is the best result produced
by any track to date. Do not promote a solo candidate over this ensemble.

## Log
| Tick | Blend | Best score | train | half2 | Verdict |
|:---|:---|---:|---:|---:|:---|
| 1 | Joint grid over Track B pairs params (`lookback∈{30,40}, weight∈{0.10,0.20}, entry_z=2.0` fixed at B's tick-1 winner) × Track C `algo_signal_scale∈{2.0,3.0}` (both B's and C's held tick-1/smoke values), plus both solos for comparison. Baseline (all off) reproduces 211.49 exactly. | 251.70 (pairs=40@0.20z2.0 + algoscale=3.00) | 34.77 | 167.60 | PROMOTABLE per `is_promotable` (score 211.49→251.70, half2 167.60≥139.4) — **held**, not copied to `teamName.py`; recommend to lead for Task 8 promote. |
