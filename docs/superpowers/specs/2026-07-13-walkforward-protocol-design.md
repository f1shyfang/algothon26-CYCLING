# Algothon 2026 — Walk-Forward Protocol Redesign

> First-principles rewrite of the research loop (rhythm, gates, promote rules).
> Keeps the scoring harness; resets the research floor to a minimal core; re-discovers
> linear regression (OLS) and pairs trading under expanding-window majority gates.

**Date:** 2026-07-13  
**Approach:** Expanding-window walk-forward (Approach 1)  
**Execution:** Parallel tracks R1 (OLS) + R2 (multi-pair) with lead merge  
**Prior art:** Supersedes gate/rhythm semantics in `AGENT_LOOP.md` and
`docs/tracks/PROTOCOL.md` for this wave. Does not delete historical iteration
logs in `STRATEGY_LOOP.md`.

---

## Goal

Raise (or honestly fail to raise) the Algothon official score
`score = μ × SR² / (SR² + 1)` on the last 250 days of `prices.txt` by:

1. Rewriting promote gates around **expanding-window walk-forward** majority votes.
2. Resetting the research floor to a **minimal core** strategy (no OLS/pairs overlays).
3. Re-running two classic families from that floor: **rolling OLS ALGO–basket** and
   **corr-screened multi-pair OLS**.

Ship to `teamName.py` only when the new gates fire and the lead reconciles.

---

## What stays vs what is rewritten

### Stays (harness / ground truth)

| Asset | Role |
|:---|:---|
| `eval.py`, `prices.txt` | Official scoring ground truth — never edit |
| `loop.py` simulator / commission / dollar limits | Must continue to bit-match `eval.py` on the official window |
| Score formula | Unchanged |
| Submission contract | `teamName.py`: numpy-only, `getMyPosition(prcSoFar) -> (51,)`, integers, no I/O |

### Rewritten for this wave

| Asset | Change |
|:---|:---|
| Promote gates in `loop.py` (`evaluate`, `is_promotable`) | Half1/half2/train retired as promote inputs; expanding folds F1–F3 + official |
| Research `Params()` defaults | Minimal core floor (see below) |
| Agent / parallel protocol docs | New walk-forward semantics (R1/R2) |
| Track docs | `docs/tracks/R1.md`, `docs/tracks/R2.md` |

Historical Tracks A/B/C/D/L1/L2 remain in the log as reject memory; they are not
reopened unless a hypothesis is absent from that memory.

---

## Architecture

### Research floor (minimal core)

`Params()` for this wave is defined as:

| Field | Value | Notes |
|:---|:---|:---|
| `lookbacks` | `(5, 20)` | Equal-weight reversal |
| `rebalance_band` | `0.195` | Production band from pre-overlay hardening |
| `regime_vol_short` / `long` | `10` / `60` | |
| `regime_threshold` / `scale` | `1.15` / `0.22` | High-vol cut |
| `algo_signal_scale` | `1.0` | Off (no ALGO amplification) |
| `ols_weight`, `mpairs_weight`, `pairs_weight`, `xs_weight`, `momentum_weight` | `0.0` | All overlays off |
| Other overlays | defaults off | |

This is **not** the current live production file (which may still hold the prior
OLS ensemble at ~265.18). Research treats the minimal core as the immutable
baseline for promote comparisons. Live `teamName.py` is left alone until a
walk-forward winner clears gates and the lead ports it.

### Parallel tracks

| Track | Family | Params prefix | Relation to floor |
|:---|:---|:---|:---|
| **R1** | Rolling OLS ALGO–basket residual | `ols_*` | Linear regression hedge on ALGO vs equal basket |
| **R2** | Corr-screened multi-pair OLS | `mpairs_*` | Pairs trading among instruments 1..50 |

**Isolation rules:**

- New / research params remain **default-off** so `Params()` stays the minimal core.
- Track agents never edit `teamName.py` or `eval.py`.
- `build_grid()` is single-owner per tick (no cross-track contamination).
- Lead-only merge into production after midday reconciliation.
- Kill rule: 3 consecutive non-promotable ticks on a track → freeze that track.

**Library policy:** Research may use numpy least-squares (preferred). Anything
promoted into `teamName.py` must be numpy-only and bit-match the harness winner
on the same `prices.txt`.

---

## Walk-forward folds & promote gates

### Data assumptions

- `prices.txt` has **500** trading days (plus header).
- Strategies may need ~60 days of warm-up (regime long window / OLS lookbacks).
- Fold score windows begin at day 201 so early history is available for signals.

### Expanding folds (fixed in harness)

| Fold | History available through | Score window | Length |
|:---|:---|:---|---:|
| **F1** | day 200 | days 201–300 | 100 |
| **F2** | day 300 | days 301–400 | 100 |
| **F3** | day 400 | days 401–500 | 100 |
| **Official** | full series | last 250 (days 251–500) | 250 |

Implementation note: each fold scores **only its OOS block** with the same
PnL/commission engine as `eval.py`. Inside the strategy, signals use only
`prcSoFar` up to the current day (no look-ahead). Fold simulation may run the
strategy from day 1 with positions/PnL accumulated, but **fold score** is
computed from daily PnL on the fold’s score window only.

### Floor evaluation

Evaluate minimal-core `Params()` on F1, F2, F3, and Official. Call that `base`.

### Robustness (`robust`)

A candidate is `robust` iff:

1. Fold score > 0 on **every** fold F1–F3, and
2. Candidate fold score **beats `base`** on **≥ 2 of 3** folds (majority).

### Promote (`promotable`)

A candidate is `promotable` iff:

1. `robust` is true, **and**
2. Official last-250 score > `base` official score, **and**
3. F3 score ≥ `0.95 × base` F3 score (do not ship a strategy that dies in the
   most recent regime).

`python loop.py --json` remains the single machine verdict (`promote: true/false`
plus fold scores). Agents must not promote on raw official score alone.

### Retired for promote decisions

The prior half1 / half2 / train split may still be printed for continuity, but
it is **not** an input to `is_promotable` in this wave.

---

## Components & signal math

### Shared helpers (unchanged intent)

- `rolling_ols_beta(y, x, lb, intercept=False) -> float` — last-`lb` OLS slope.
- `spread_z(spread, lb) -> float` — z-score of last point vs last-`lb` window.
- Default-off invariance: `ols_weight=0` and `mpairs_weight=0` leave scores
  identical to the minimal core.

### Track R1 — `ols_*`

| Param | Research default | Role |
|:---|:---|:---|
| `ols_lookback` | 40 (off until weight > 0) | Window for β and z |
| `ols_weight` | 0.0 | Blend into base signal |
| `ols_entry_z` | 2.0 | Trade only if \|z\| ≥ threshold |
| `ols_intercept` | False | Optional intercept in OLS |

**Signal:** On ALGO log-price vs equal-weight basket of instruments 1..n, fit β,
form `spread = ALGO − β · basket`, z-score, mean-revert when \|z\| ≥ entry.
Blend: `(1 − w) · signal + w · ols_sig` when active.

**Mode for this wave:** Replace-only relative to any legacy `pairs_*` overlay
(`pairs_weight` stays 0). Do not run additive pairs+OLS without labeling it as a
separate hypothesis.

### Track R2 — `mpairs_*`

| Param | Research default | Role |
|:---|:---|:---|
| `mpairs_lookback` | 60 | Screen + OLS + z window |
| `mpairs_weight` | 0.0 | Blend weight |
| `mpairs_top_k` | 5 | Number of pairs |
| `mpairs_entry_z` | 2.0 | Per-pair entry |
| `mpairs_min_corr` | 0.65 | Abs corr floor (must be ≤ empirical max ~0.77) |

**Screen:** Among instruments 1..50 (ALGO excluded), rank pairs by \|corr\| of
log returns over lookback; keep `|corr| ≥ min_corr`; take top-k.

**Signal:** Per pair, OLS β, spread z-score; if \|z\| ≥ entry, long/short in
hedge-ratio space; average active pairs; blend only when the overlay is active
(no dilution when zero pairs fire).

**Reject-memory carry-forward:** Prior L2 tick failed because `min_corr` was
set above empirical pairwise corr — R2 tick-1 grids must use
`min_corr ∈ {0.55, 0.65, 0.70}`.

---

## Protocol rhythm

```
Morning assign → Parallel R1+R2 ticks → Midday merge → Evening log
```

### Morning assign (lead)

1. Confirm research floor: `Params()` is minimal core; walk-forward baseline
   scores are recorded.
2. Read each track’s reject memory and “next hypothesis” note.
3. Assign exactly one untried hypothesis per track.

### Parallel ticks (agents)

- One iteration of the (rewritten) agent loop: hypothesis → own `build_grid()` →
  `python loop.py --sweep` → `python loop.py --json` → log → stop.
- Touch only the assigned Params prefix.
- Never edit `teamName.py` / `eval.py`.

### Midday merge (lead)

| Verdicts | Action |
|:---|:---|
| All `promote: false` | No production change; update track logs only |
| Exactly one `promote: true` | Lead promote procedure for that winner |
| Two `promote: true` | Keep higher official score that still passes F3 ≥ 0.95×base F3; park the other for an optional ensemble tick |

### Lead promote procedure

1. Confirm `promote: true` via `loop.py --json` before touching production.
2. Port winning constants + numpy OLS/pairs logic into `teamName.py`.
3. Sync `loop.py` `Params` defaults to the new production floor (overlays that
   won become defaults; the other track’s weight stays 0 unless ensembled).
4. Verify: `python eval.py` score matches harness official score; tests green;
   numpy-only audit.
5. Refresh `CYCLING.py` if that remains the submission alias.
6. Log Iteration entry in `STRATEGY_LOOP.md` and update track doc status.

### Kill rule

> 3 consecutive non-promotable ticks on a track → freeze that track.

Consecutive is per-track. Revive only with a genuinely new hypothesis family
absent from that track’s reject memory.

### Stopping criteria (wave-level)

Declare the wave **DONE** when:

- Both tracks are frozen or their hypothesis banks are exhausted, **or**
- A ±30% sensitivity sweep around the current research/production floor shows a
  stable plateau with no promotable neighbour under the walk-forward gates, **or**
- The user stops / deadline hits.

---

## First grids

| Track | Tick-1 hypothesis | Grid |
|:---|:---|:---|
| **R1** | Rolling OLS ALGO–basket residual mean-reverts vs minimal core | `ols_lookback∈{30,40,60}` × `ols_weight∈{0.10,0.20,0.30}` × `ols_entry_z∈{1.5,2.0,2.5}`; `ols_intercept=False` |
| **R2** | Corr-screened multi-pair OLS adds orthogonal mean-reversion | `mpairs_lookback∈{40,60}` × `top_k∈{3,5}` × `weight∈{0.10,0.20}` × `entry_z∈{1.5,2.0}` × `min_corr∈{0.55,0.65,0.70}` |

Optional later: if both are promotable-held (or one promoted and the other
strong), one joint `ols_* + mpairs_*` ensemble grid before a final promote —
same spirit as prior Track D, but judged with walk-forward gates.

---

## Testing & success criteria

| Check | Requirement |
|:---|:---|
| Floor invariance | `Params()` = minimal core; `ols_weight=0` / `mpairs_weight=0` unchanged |
| Fold engine | F1–F3 scores are deterministic and use the same PnL engine as official |
| Promote | `is_promotable` implements majority + official ↑ + F3 gate exactly as above |
| Post-promote | `test_teamName.py` green + `eval.py` matches harness official + numpy-only |
| Production safety | Live `teamName.py` unchanged until lead promote |

**Success:**

- Harness exposes fold scores + new promote verdict.
- At least one real R1 and one real R2 sweep tick logged.
- Production score never changes without lead promote.
- If promote occurs, official ↑ and walk-forward gates passed.
- If both tracks freeze, STRATEGY_LOOP records clear reject lessons.

---

## Risks & mitigations

| Risk | Mitigation |
|:---|:---|
| Overlapping official window with F2/F3 | Official remains ship gate; majority across F1–F3 still required |
| Early-fold warm-up / short history for OLS | Folds start at day 201; lookbacks capped in grids |
| R2 empty-pair dilution | Gate blend when overlay inactive; min_corr ≤ empirical corr |
| Confusion with old half1/half2/train | Print legacy metrics optionally; exclude from promote |
| Live production vs research floor drift | Document both scores; only lead promote edits `teamName.py` |
| Re-fitting prior OLS winner by memory | Re-discover from core; old L1 params are candidates, not defaults |

---

## Out of scope

- Reopening rejected xs / momentum / EMA hypotheses
- Editing `eval.py` or `prices.txt`
- Nested locked-holdout (Approach 3) or rolling fixed-length folds (Approach 2)
- Auto leaderboard submission
- Non-numpy dependencies in `teamName.py`

---

## Deliverables

1. Spec (this document).
2. Implementation plan under `docs/superpowers/plans/`.
3. `loop.py`: expanding-fold evaluation + rewritten `is_promotable`; `Params()`
   reset to minimal core for the research wave.
4. Rewritten agent/protocol docs + `docs/tracks/R1.md`, `docs/tracks/R2.md`.
5. First R1/R2 ticks (after plan execution), then lead merge if gates fire.
6. Promote path: `teamName.py` + Params sync + `CYCLING.py` refresh only on
   lead promote.

---

## Design decisions (locked)

1. Protocol redesign from first principles; keep scoring harness only.
2. Research floor = minimal core (reversal + band + regime); re-discover OLS/pairs.
3. Parallel R1 + R2 with lead merge.
4. Expanding-window folds F1–F3; majority (≥2/3) beat floor; official ↑; F3 ≥ 0.95×base F3.
