# Day 4 — Freeze, Stress, Submit Checklist

> Task 9 of `docs/superpowers/plans/2026-07-11-algothon-parallel-tracks.md`.
> Production baseline after Task 8 (Iteration 29): **score 251.70**
> (5d/20d vol-standardised reversal + band 0.195 + vol-regime cut
> 10/60@1.15→0.22 + Track B pairs overlay 40@0.20 z=2.0 + Track C ALGO
> scale 3.0). See `STRATEGY_LOOP.md` Iteration Log for full derivation.

---

## 1. Freeze

- [x] No new strategy families introduced this task.
- [x] `teamName.py` signal logic unchanged from Iteration 29's promote.
- [x] Only activity: stress notes, contract audit, and packaging.

---

## 2. Stress checks (research only, no `eval.py` changes)

- [x] **Commission stress** — prior finding (Iteration 9, still applicable):
  strategy stayed profitable at 2x fees (Sharpe 1.69) and 5x fees
  (Sharpe 1.42) under the pre-ensemble strategy. The Track B/C overlays
  added in Task 8 are lower-turnover, event-triggered (pairs fires only
  when `|z| >= 2.0`) or a pure per-instrument scale (ALGO), so commission
  sensitivity is not expected to have worsened; a full re-run of the 2x/5x
  fee sweep on the current ensemble was **not** repeated this task (no
  signal change occurred that would invalidate the qualitative
  conclusion), consistent with the Task 9 instruction to only re-verify
  when a test fails.
- [x] **Synthetic ±5% overnight gap smoke test** (quick script, this task):
  ran `getMyPosition` on the full `prices.txt` history with the final
  day's prices shocked by +5% and -5% and confirmed for both directions:
  - output stays finite and integer-typed,
  - ALGO notional stays within the $100,000 cap,
  - all other instruments' notional stays within the $10,000 cap.
- [x] **Commission-aware turnover** — the 0.195 rebalance band (unchanged
  from prior iterations) still suppresses sub-threshold position churn;
  the pairs overlay only engages on rare, strongly-stretched
  ALGO-vs-basket divergences (`entry_z=2.0`), so it does not add
  meaningful daily turnover in the common case.
- [x] **Position-limit clipping** — `getMyPosition` clips to
  `dollar_limits / current_prices` (10K default, 100K ALGO) before
  returning, independent of `eval.py`'s own clip, matching
  `STRATEGY_LOOP.md`'s Phase 4 warning.

**Conclusion:** no regressions found; no code changes required.

---

## 3. Submission contract audit

Commands run and results (this task, Day 4):

```bash
python -m unittest test_teamName.py -v
# Ran 8 tests in 0.078s -- OK (all 8 pass)

python eval.py 2>&1 | tail -8
# Day 500 value: 74512.80 todayPL: $1377.16 $-traded: 35712366 return: 0.00209
# =====
# mean(PL): 298.1
# return: 0.00209
# StdDev(PL): 2022.24
# annSharpe(PL): 2.33
# totDvolume: 35712366
# Score: 251.70

rg -n "^(import|from) " teamName.py
# 1:import numpy as np
```

Checklist:

- [x] Score positive and matches last promote (251.70, exact)
- [x] Only `numpy` imported — allowed for grading (`requirements-dev.txt`
      permits numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib)
- [x] No file I/O / network calls in `teamName.py`
- [x] `getMyPosition(prcSoFar)` signature exact; returns shape `(51,)`,
      integer dtype, finite values, within $10K/$100K dollar limits
      (verified directly against `prices.txt`, not just via `eval.py`)

---

## 4. Package

```bash
cp teamName.py CYCLING.py
```

- Team folder is `algothon26-CYCLING` → submission filename is
  **`CYCLING.py`** (README says copy `teamName.py` → `<YourTeamName>.py`;
  if the wiki's registered team name ever differs from `CYCLING`, rename
  the copy to match the wiki/registration, not the folder).
- `CYCLING.py` verified byte-identical to `teamName.py` (`diff` clean) and
  independently imports + runs `getMyPosition` with correct output shape.
- `eval.py` and `test_teamName.py` still point at `teamName` (per README,
  `eval.py` imports from `teamName` by default during development) — no
  changes needed there for the submission copy to be valid.

---

## 5. Submit (not performed by this task)

This task does **not** submit to the live leaderboard. When ready:

1. Re-run `python eval.py` one final time against the *exact* file being
   submitted (rename-and-rerun) to confirm score parity.
2. Zip only the algorithm file per the
   [Submission Guide](https://wiki.algothon.au/submission/) (`CYCLING.py`,
   plus `requirements.txt` only if extra packages were used — not needed
   here, numpy only).
3. Submit via <https://www.algothon.au/leaderboard> (limit: ≤1/day).
4. Record submission timestamp + claimed score in `STRATEGY_LOOP.md`
   immediately after submitting.

**No submission has been made as part of Task 9.** This is documentation
only, per instructions.

---

## 6. 16 Jul restart playbook

When the General Round `prices.txt` drops (expected 16 Jul):

1. **Replace data:** overwrite `prices.txt` with the new General Round
   file (never edit `eval.py`).
2. **Re-baseline:** run `python eval.py 2>&1 | tail -8` to get the new
   score under the *current* frozen strategy — this is the new floor,
   not necessarily 251.70 (price distribution will differ).
3. **Sync harness:** update `loop.py`'s `Params` defaults if they've
   drifted from `teamName.py` (they should already match post-Task 8);
   re-run `python loop.py 2>&1 | tail -20` and confirm it matches the new
   `eval.py` score to within 0.01. Fix `loop.py`, never `eval.py`, on
   mismatch.
4. **Reset track logs, keep reject memory:** in each `docs/tracks/{A,B,C,D}.md`,
   clear the "Log" table's *tick counters* for the new data but **do not
   delete reject memory** — hypotheses already disproven on the prior
   data (e.g. Track A's demean+std xs overlay, Track C's ALGO-cap cuts)
   are still informative priors, not automatically valid rejects on new
   data, but should be re-tried cautiously rather than assumed dead.
5. **Restart Tasks 6–8** (`docs/superpowers/plans/2026-07-11-algothon-parallel-tracks.md`):
   relaunch parallel tracks A/B/C against the new floor, apply the same
   kill rule (3 consecutive non-promotable ticks → freeze), and use
   Task 8's promote procedure for any robust winner.
6. **Re-run this Task 9 checklist** (freeze → stress → audit → package)
   before any new submission on the General Round data.
