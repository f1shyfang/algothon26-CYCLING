# Parallel Agent Execution Protocol

> Operational rhythm for running research tracks in parallel against the shared
> `loop.py` research harness, with a lead agent/human reconciling promotes.
> Companion to `AGENT_LOOP.md` (per-tick protocol for a single agent).

---

## Wave: Walk-forward R1/R2 (current)

**Research floor:** minimal-core `Params()` → `python loop.py` scores **211.49**
(all overlays off: 5d/20d reversal, band=0.195, regime 10/60@1.15→0.22).
Live `teamName.py` / `eval.py` may still show **265.18** until the lead
promotes — use `loop.py` for floor checks during research ticks.

**Promote gates** (from `python loop.py --json`):
- Majority of walk-forward folds F1–F3 beat baseline fold scores
- Official score strictly above baseline
- F3 ≥ 0.95× baseline F3
- Do **not** promote on official score alone

| Track | Prefix | Doc |
|:---|:---|:---|
| **R1** — rolling OLS ALGO–basket | `ols_*` | `docs/tracks/R1.md` |
| **R2** — multi-pair OLS | `mpairs_*` | `docs/tracks/R2.md` |

### Daily rhythm

```
Morning assign  →  Parallel ticks  →  Midday merge  →  Evening log
     (lead)          (R1, R2)          (lead)          (lead)
```

**Morning assign (lead)**
1. Confirm research floor: `python loop.py` → official score **211.49** (not
   `eval.py` until after a promote).
2. Read each `docs/tracks/{R1,R2}.md` reject memory and prior tick log.
3. Assign exactly **one** untried hypothesis per track — never re-assign a
   hypothesis already in a track's reject memory.

**Parallel ticks (agents R1, R2)**
- Each agent runs **one iteration** of `AGENT_LOOP.md` end to end: form
  hypothesis → extend `build_grid()` for its own `Params` prefix only →
  `python loop.py --sweep` → `python loop.py --json` → log → stop.
- Touch only your track's `Params` fields in `build_grid()`; leave all other
  overlays at default (off).
- No agent edits `teamName.py` or `eval.py`. Research only.

**Midday merge (lead)**
1. Collect each track's `python loop.py --json` verdict.
2. Decision:
   - All `promote: false` → no production change; update track docs/log only.
   - Exactly one `promote: true` → proceed to promote procedure for that winner.
   - Two or more claim `promote: true` → keep the higher official score that
     still passes all walk-forward gates; park the other in reject memory.
3. Apply the **kill rule** (below).

**Evening log (lead)**
- Append one summary entry to `STRATEGY_LOOP.md`'s Iteration Log covering both
  ticks run that day.

### Kill rule

> **3 consecutive non-promotable ticks on a track → freeze that track.**

- Consecutive is per-track, not global.
- On freeze: set `**Status:** frozen` in `docs/tracks/{R1,R2}.md` and reassign
  the agent or idle until a genuinely new hypothesis family emerges.

### Lead-only promote rule

> **Only the lead merges into `teamName.py` — never a track agent.**

- Track agents stop at logging the verdict (`promote: true/false` + F1/F2/F3 +
  official numbers) in their track doc and `STRATEGY_LOOP.md`.
- A `promote: true` verdict is necessary but not sufficient — the lead still
  reconciles across tracks at midday merge before touching `teamName.py`.
- Promoting requires: confirm verdict → port winning `Params` into
  `teamName.py` → `python eval.py` matches `loop.py` winner score →
  `python -m unittest test_teamName.py -v` all green → sync `loop.py`
  `Params` defaults → log + commit. Any `eval.py`/`loop.py` mismatch reverts
  the promote and is logged as harness drift.

### Agent launch cheat-sheets

**Track R1**

```text
Track R1 only. Read AGENT_LOOP.md + docs/tracks/R1.md.
One hypothesis. Touch only ols_* Params in loop.py build_grid.
Never edit teamName.py/eval.py. Run --sweep and --json. Log and stop.
```

**Track R2**

```text
Track R2 only. Read AGENT_LOOP.md + docs/tracks/R2.md.
One hypothesis. Touch only mpairs_* Params in loop.py build_grid.
Never edit teamName.py/eval.py. Run --sweep and --json. Log and stop.
```

---

## Historical — Parallel Agent Execution Protocol (Day 1–2, Tracks A/B/C)

> *Superseded by walk-forward R1/R2 above. Kept for context.*

> Operational rhythm for running Tracks A/B/C in parallel against the shared
> `loop.py` research harness, with a lead agent/human reconciling promotes.
> Companion to `AGENT_LOOP.md` (per-tick protocol for a single agent) and
> `docs/superpowers/plans/2026-07-11-algothon-parallel-tracks.md` (full plan).

---

## Daily rhythm

```
Morning assign  →  Parallel ticks  →  Midday merge  →  Evening log
     (lead)          (A, B, C)          (lead)          (lead)
```

### Morning assign (lead)
1. Confirm the floor hasn't drifted: `python eval.py 2>&1 | tail -3` must match
   the baseline recorded in `STRATEGY_LOOP.md` (currently **211.49**).
2. Read each `docs/tracks/{A,B,C}.md` reject memory and "next hypothesis" note
   from the prior tick's log entry.
3. Assign exactly **one** untried hypothesis per track for this round — never
   re-assign a hypothesis already in a track's reject memory.
4. Hand each agent the matching cheat-sheet below (also in the plan doc).

### Parallel ticks (agents A, B, C)
- Each agent runs **one iteration** of `AGENT_LOOP.md` end to end: form
  hypothesis → extend `build_grid()` for its own `Params` prefix only →
  `python loop.py --sweep` → `python loop.py --json` → log → stop.
- Agents must leave every other track's `Params` fields at their default
  (off) value in `build_grid()` — `build_grid()` is single-owner per tick to
  avoid cross-track contamination (see fix `2f5e4ca`, which made it
  Track-C-only for Task 5). Whichever track is actively sweeping owns
  `build_grid()` for the duration of its tick.
- No agent edits `teamName.py` or `eval.py`. Research only.

### Midday merge (lead)
1. Collect each track's `python loop.py --json` verdict.
2. Decision:
   - All `promote: false` → no production change; update track docs/log only.
   - Exactly one `promote: true` → proceed to `Task 8` (promote procedure) for
     that winner only.
   - Two or more claim `promote: true` → keep the higher official score that
     still passes `half2 ≥ 0.95 × baseline half2`; park the other candidate in
     `docs/tracks/D.md` under Candidates for the Day-3 ensemble.
3. Apply the **kill rule** (below).

### Evening log (lead)
- Append one summary line/entry to `STRATEGY_LOOP.md`'s Iteration Log
  covering all three ticks run that day.
- Optional: one leaderboard submission if the live stage accepts entries
  (≤1/day).

---

## Kill rule

> **3 consecutive non-promotable ticks on a track → freeze that track.**

- Consecutive is per-track, not global — Track A's rejects don't count
  against Track B or C.
- On freeze: set `**Status:** frozen` in that track's `docs/tracks/{X}.md`
  header and reassign the agent to (a) a different hypothesis-bank family the
  track hasn't tried, or (b) idle, pending the Day-3 ensemble (Track D).
- A track can be revived later in the run if a genuinely new hypothesis
  family emerges (not a retry of anything in its reject memory).

---

## Lead-only promote rule

> **Only the lead merges into `teamName.py`, and only via the `Task 8`
> promote procedure — never a track agent.**

- Track agents' job stops at "log the verdict" (`promote: true/false` +
  numbers) in their own `docs/tracks/{X}.md` and in `STRATEGY_LOOP.md`.
- A `promote: true` verdict from `loop.py --json` is necessary but not
  sufficient — the lead still reconciles across tracks at midday merge (see
  above) before touching `teamName.py`.
- Promoting requires, in order: confirm verdict → port winning `Params` into
  `teamName.py` constants/logic → `python eval.py` matches `loop.py`'s winner
  score → `python -m unittest test_teamName.py -v` all green → update
  `loop.py` `Params` defaults to the new floor → log + commit. Any mismatch
  between `eval.py` and `loop.py` reverts the promote and is logged as harness
  drift, not a strategy result.

---

## Agent launch cheat-sheets

**Track A**

```text
Track A only. Read AGENT_LOOP.md + docs/tracks/A.md.
One hypothesis. Touch only xs_* Params in loop.py build_grid.
Never edit teamName.py/eval.py. Run --sweep and --json. Log and stop.
```

**Track B**

```text
Track B only. Read AGENT_LOOP.md + docs/tracks/B.md.
One hypothesis. Touch only pairs_* Params in loop.py build_grid.
Never edit teamName.py/eval.py. Run --sweep and --json. Log and stop.
```

**Track C**

```text
Track C only. Read AGENT_LOOP.md + docs/tracks/C.md.
One hypothesis. Touch only algo_signal_scale / algo_hedge_weight in build_grid.
Never edit teamName.py/eval.py. Never retest $50K ALGO caps. Run --sweep/--json. Log and stop.
```

---

## Current Day-1 status summary

Baseline in `teamName.py` remains **211.49** (5d/20d vol-standardised
reversal, band=0.195, regime 10/60@1.15→scale 0.22). No promotes have been
applied to production yet — Day 1 was pure parallel research per Tasks 3–5.

| Track | Tick 1 result | Verdict | Status |
|:---|:---|:---|:---|
| **A** — cross-sectional (`xs_*`) | Best xs=10@0.20+algo → 185.00 vs baseline 211.49 (all 8 variants 154.64–185.00) | **REJECTED** | active, 1 consecutive reject |
| **B** — pairs (`pairs_*`) | pairs=40@0.20 z=2.0 → 219.30 (+7.81), half2=148.12 | **PROMOTABLE — HELD** for lead reconciliation | active, held candidate |
| **C** — ALGO-centric (`algo_*`) | algo_signal_scale=2.0 → 228.65 (+17.15), half2=155.53; `algo_hedge_weight` monotonically bad (dead arm) | **PROMOTABLE — HELD**; grid-boundary result (scale=2.0 was the edge of the screened range) — extend before promoting | active, held candidate |

Two tracks (B, C) are holding promotable candidates simultaneously. Per the
midday-merge rule above, the lead has **not** promoted either yet — Task 6
is scaffolding/protocol only, and the plan's own Task 5 note flags that
Track C's win is a grid-boundary result that needs extension before any
promote decision is made. `build_grid()` in `loop.py` is currently
Track-C-only (fix `2f5e4ca`).

---

## Day-2 hypothesis suggestions (next tick per track)

**Track A** — try a different `xs` formulation, or freeze if no new idea
survives scrutiny (already 1 reject):
- Rank-based cross-sectional signal instead of demean+std (the rejected
  family), e.g. buy bottom decile / sell top decile by 5d or 10d return.
- If no genuinely different formulation is available, this counts as the
  track's first reject toward the kill rule — do not repeat the demean+std
  family at a different lookback/weight, that's still "the same idea."

**Track B** — sensitivity sweep around the held candidate, not a new family:
- Grid: `pairs_entry_z ∈ {1.75, 2.0, 2.25}` × `pairs_lookback ∈ {30, 40, 50}`
  (weight held at 0.20, the tick-1 winner).
- Goal: confirm pairs=40@0.20z2.0 is a local peak (or find a better
  neighbour) before the lead reconciles B against C.

**Track C** — extend the grid boundary before any promote:
- Grid: `algo_signal_scale ∈ {2.5, 3.0, 3.5, 4.0}` (hedge arm stays dead per
  reject memory — do not re-test `algo_hedge_weight`).
- Goal: find where gains flatten or half2/train start degrading, since
  unbounded amplification eventually saturates against ALGO's $100K cap.
- A quick research-only smoke of exactly this grid (see `docs/tracks/C.md`)
  already suggests the answer: scale=3.0 scored highest (234.80) but the
  gain is **not** monotonic past 2.0 — 2.5/3.0/3.5/4.0 all land in a narrow
  231–235 band with 3.5 dipping below its neighbours. Track C's real Day-2
  tick should re-run this properly (fine grid ~2.25–3.25) and log it as a
  formal tick to settle the local optimum, since the smoke run was not
  logged as a tick and made no `loop.py` commit.

---

## Recommended next promote candidate (after Day-2 suggestions)

Track C's `algo_signal_scale=2.0` currently has the larger score gain
(228.65 vs B's 219.30), but per its own reject-memory note it's a
**grid-boundary result** — not yet a settled local optimum. A quick
research-only smoke of `{2.5, 3.0, 3.5, 4.0}` (logged in `docs/tracks/C.md`,
not a formal tick) found scale=3.0 higher still (234.80) but with a
non-monotonic, narrow-band (231–235) pattern past 2.0 — consistent with a
local plateau rather than an open boundary. That's encouraging (C likely has
more headroom than B) but not yet a settled, formally-logged tick.

Recommendation: run Track C's Day-2 tick properly (fine grid ~2.25–3.25,
logged as a real tick with a `docs/tracks/C.md` Log row) to settle the local
optimum; if it still passes robustness guards (`half2`/`train` intact) at
the new optimum, promote **C**. Track B's candidate is already a
fully-screened local result from its tick-1 grid and remains the safer
near-term promote if a decision is needed before C's Day-2 tick completes.

---

## Day-4 freeze (Task 9)

**Status: FROZEN for submission.** Both held candidates above (Track B,
Track C) were superseded by the Track D ensemble (Iteration 28) and
promoted to production by the lead (Iteration 29) *before* Day 4 began —
see `STRATEGY_LOOP.md` Iterations 28–29. Production baseline entering
Day 4 is **251.70** (pairs=40@0.20z2.0 + algo_signal_scale=3.0 on top of
the Day-2 band/regime floor), not the 211.49/228.65/219.30 figures
referenced earlier in this document, which are now historical.

Day-4 rules in effect:

- No new strategy families (Task 9 Step 1). Track agents stop taking new
  ticks; `teamName.py` signal logic does not change unless a test fails.
- Stress checks and the submission contract audit are research/verification
  only — see `docs/tracks/SUBMIT.md` for the full Day-4 checklist and
  results (8/8 tests pass, `eval.py` score = 251.70, only `numpy`
  imported, signature/shape/limits verified directly).
- Submission packaging (`CYCLING.py`, the registered-team-name copy of
  `teamName.py`) is prepared but **not submitted** to the live
  leaderboard by this task — see `docs/tracks/SUBMIT.md` §5 for the
  submit procedure and §6 for the 16 Jul General Round restart playbook.

---

## Wave 2 — OLS / Multi-pair (L1 + L2)

**Floor:** 251.70 (pairs+ALGO-scale ensemble). Spec:
`docs/superpowers/specs/2026-07-12-ols-mpairs-tracks-design.md`

| Track | Prefix | Notes |
|:---|:---|:---|
| L1 | `ols_*` | Prefer replace mode (`pairs_weight=0`) when sweeping |
| L2 | `mpairs_*` | Instruments 1..50 only; corr screen; no daily cointegration |

Same daily rhythm, kill rule, and lead-only promote as Wave 1.
Prior tracks A/B/C/D are historical — do not reopen rejected hypotheses.

**Wave 2 tick-1 status (for lead):**
| Track | Best official | half2 | promote? |
|:---|---:|---:|:---|
| L1 | 265.18 | 211.35 | **PROMOTED** (Task 6 — ols=30@0.20z1.5 replace) |
| L2 | 221.94 | 146.21 | false (corr floor too high; blend gate fixed) |

**Post-Task-6 floor:** **265.18**. Task 7 ensemble skipped (L2 weak). `Params()` /
`teamName.py` now use OLS replace (`pairs_weight=0`, `ols_weight=0.20`).
