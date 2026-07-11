# Algothon 2026 — Parallel Strategy Tracks Design

> Competition plan for Susquehanna x UNSW Algothon 2026 (team CYCLING).
> Approved brainstorming outcome: learning-first exploration (new strategy families)
> via parallel agents, while maximising leaderboard score under a robust promote gate.

**Date:** 2026-07-11  
**Horizon:** 4 days (ends just before General Round dataset release on Thu 16 Jul)  
**Approach:** Parallel strategy tracks (Approach 1)

---

## Goal

Raise (or protect) the official Algothon score on the last 250 days of `prices.txt`:

`score = μ × SR² / (SR² + 1)`, where `SR = √250 · μ/σ` of daily PnL.

Do this by exploring multiple untried strategy families in parallel with agents, while
keeping the current production strategy as a hard floor until a candidate clears the
existing overfitting guards.

**Current production baseline:** volatility-standardised 5d/20d time-series reversal,
19% rebalance band, high-vol regime cut (10d/60d @ 1.15 → 50% exposure). See
`teamName.py` and `STRATEGY_LOOP.md`.

---

## Architecture

**Floor:** `teamName.py` is production. Nothing ships unless `python loop.py --json`
returns `promote: true` (official score improvement plus train / half1 / half2 guards).

**Tracks:**

| Track | Role | Strategy family | Goal |
|:---|:---|:---|:---|
| **P0** | Lead / human | Production hardening | Protect baseline; small robust tweaks only |
| **A** | Agent | Cross-sectional mean reversion | Diversify time-series edge |
| **B** | Agent | Pairs / cointegration | Low-correlation PnL vs reversal |
| **C** | Agent | ALGO-centric / hedge | Exploit $100K limit + cheaper fees |
| **D** | Agent | Ensemble blender | Combine only tracks that beat standalone noise |

**Day split:**

| Days | Focus |
|:---|:---|
| 1–2 | Explore A/B/C in parallel (research only; no production edits unless promote) |
| 3 | Promote winners; build D from uncorrelated survivors |
| 4 | Freeze, stress, package submission, leaderboard submit if stage open |

**Merge rule:** At most one promote per checkpoint. If two tracks promote the same day,
keep the higher official score that still passes half2/train guards. The other becomes
an ensemble candidate for Day 3 (Track D).

**Calendar note:** General Round dataset releases **Thu 16 Jul**; General Round closes
**Thu 30 Jul**. This 4-day window banks a robust floor and a reusable playbook; on new
`prices.txt`, re-baseline and restart tracks A–D without discarding the iteration log.

---

## Agent protocol

Each parallel agent runs **one iteration per tick**, following the discipline in
`AGENT_LOOP.md`, with track isolation.

### Hard rules

1. Never edit `eval.py`, `prices.txt`, or scoring / simulator ground truth in `loop.py`.
2. Never edit `teamName.py` unless the promote gate fired for that track’s winner.
3. One hypothesis per tick; log every reject in `STRATEGY_LOOP.md` (or a per-track section).
4. Research only via `Params` / `build_grid` / optional scratch strategy functions —
   default-off so the baseline is unchanged when the new field is unused.
5. Every number in the log must come from a real run (`python loop.py --sweep` / `--json`).
6. Submission-safe production: only packages allowed for grading; no file I/O, no network;
   signature exactly `getMyPosition(prcSoFar) -> np.ndarray` of shape `(51,)`, integer
   positions within dollar limits.

### Per-tick checklist

1. Read iteration log + current production constants.
2. State one hypothesis using the template in `STRATEGY_LOOP.md`.
3. Add candidates to the grid → run sweep.
4. Run `python loop.py --json` → promote or reject.
5. If promote: update `teamName.py` → confirm `python eval.py` and
   `python -m unittest test_teamName.py -v`.
6. Append log; propose next hypothesis; stop.

### Conflict avoidance

- Agents A/B/C work on **separate Params fields / strategy functions** (no shared
  mutable knobs that would invalidate each other’s grids).
- Only **P0 / lead** merges promotes into `teamName.py`.
- If two agents both want promote: lead compares JSON verdicts; ships the better robust
  winner; parks the other for Track D.

### Parallel cadence

Roughly 3 agents × 6–8 ticks/day on Days 1–2 ≈ 40–50 hypotheses screened without
touching production until something clears the gate.

---

## Four-day schedule

### Daily rhythm (~90–120 min human + agents)

1. **Morning:** assign one hypothesis each to A/B/C (or D on Day 3).
2. **Agents:** run ticks in parallel.
3. **Midday:** lead reviews `--json` verdicts; merge at most one promote.
4. **Evening:** update log; pick tomorrow’s hypotheses; optional daily leaderboard
   submit (competition limit: one algorithm submission per day).

### Day 1 — Bootstrap + screen

- Spin tracks A, B, C.
- Success: each track has ≥3 scored hypotheses logged; production untouched unless a
  candidate promotes.

### Day 2 — Double down

- Continue A/B/C; kill dead tracks early.
- Success: ≥1 track with train + half2 not collapsing; drop any track with 3 straight
  rejects (kill criteria).

### Day 3 — Promote + ensemble

- Promote any cleared winners into production (lead only).
- Track D blends survivors with low PnL correlation and positive standalone score.
- Success: if promote, new baseline confirmed by `eval.py` + tests; else keep current
  baseline and document why.

### Day 4 — Freeze + submit ops

- No new strategy families.
- Stress retained strategy (e.g. 2× fees, synthetic gaps) as already practiced in the
  strategy loop.
- Package submission; submit if the live stage accepts entries.
- Write restart playbook for 16 Jul data refresh.

### Kill criteria (per track)

Three consecutive non-promotable ticks → freeze that family; reassign the agent to
another idea or to Track D.

---

## Artifacts

| Artifact | Purpose |
|:---|:---|
| `teamName.py` | Production strategy only (promote-gated) |
| `STRATEGY_LOOP.md` | Shared iteration log + backtest table |
| `AGENT_LOOP.md` | Agent iteration protocol |
| `loop.py` | Research harness, grids, promote verdict |
| `eval.py` | Official local scorer (do not edit) |
| `test_teamName.py` | Contract / behavioural tests |
| `docs/tracks/{A,B,C,D}.md` (optional) | Per-track hypothesis banks if main log is noisy |
| `<YourTeamName>.py` or wiki-required name | Submission copy (Day 4 / submit day only) |

---

## Day-4 / submission checklist

- [ ] `python eval.py` matches last promoted score
- [ ] `python -m unittest test_teamName.py -v` all green
- [ ] Imports ⊆ packages available at grading (`requirements-dev.txt` / wiki list)
- [ ] No file I/O / network / external data in the submission file
- [ ] Signature `getMyPosition(prcSoFar)` → shape `(51,)`, ints, within $10K / $100K (ALGO)
- [ ] Package per [Submission Guide](https://wiki.algothon.au/submission/)
- [ ] Submit via [leaderboard](https://www.algothon.au/leaderboard) (≤1/day)
- [ ] On **16 Jul** new data: re-run `eval.py`, re-baseline, restart tracks A–D; keep the log

---

## Risks and mitigations

| Risk | Mitigation |
|:---|:---|
| Overfitting from parallel search | Promote gate (train + halves); one promote per checkpoint |
| Agents collide on `teamName.py` | Only lead merges; agents research in `loop.py` Params |
| Wasting Days 1–2 on dead ends | Kill after 3 consecutive rejects; reassign agent |
| New data (16 Jul) resets edge | Treat Days 1–4 as playbook; re-run same process on new prices |
| Ensemble dilutes edge | D only mixes tracks with low PnL corr and standalone positive score |

---

## Definition of done (this 4-day window)

1. Best robust `teamName.py` frozen under the promote gate.
2. Submission file ready and checklist complete.
3. Iteration log records what worked and what failed (including rejects).
4. Clear restart plan for General Round `prices.txt` on 16 Jul.

---

## Out of scope (this plan)

- Editing `eval.py` or changing official scoring semantics.
- Blind hyperparameter fishing without a stated hypothesis.
- Shipping ensembles that fail the promote gate.
- Full ML stack beyond packages allowed at grading time.
