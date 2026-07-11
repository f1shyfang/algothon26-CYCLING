# Algothon 2026 — OLS / Multi-Pair Research Tracks Design

> Follow-on research wave after the Day-1–4 parallel tracks (A/B/C/D) that
> produced the current production ensemble (score **251.70**).
> Approved brainstorming: explore winner-style regression strategies as new
> research tracks without risking the production floor until promote.

**Date:** 2026-07-12  
**Approach:** Parallel tracks (Approach 1) — Track L1 + Track L2  
**Floor:** `teamName.py` production (5d/20d reversal + band 0.195 + regime
10/60@1.15→0.22 + pairs 40@0.20 z=2.0 + ALGO scale 3.0)

---

## Goal

Raise the official Algothon score on the last 250 days of `prices.txt`
(`score = μ × SR² / (SR² + 1)`) by testing two classic winner-style families:

1. **L1 — Rolling OLS ALGO–basket hedge** (upgrade of equal-weight β≈1 pairs)
2. **L2 — Multi-pair OLS mean reversion** (corr-screened pairs with OLS hedge ratios)

Keep the current production strategy as a hard floor until a candidate clears
the existing overfitting guards in `loop.py`.

**Shortlist context:** Winner patterns surveyed included OLS hedge ratios,
multi-pair corr→mean-rev, rolling market beta residuals, lead-lag, and PCA
residuals. Chosen after C→A shortlist: **L1 + L2**. Deferred: L3 (beta residual,
similar to rejected Track A), L4 (lead-lag), L5 (PCA).

---

## Architecture

**Floor stays frozen:** Nothing ships to `teamName.py` unless
`python loop.py --json` returns `promote: true` and the lead runs the promote
procedure.

| Track | Params prefix | Strategy family | Relation to production |
|:---|:---|:---|:---|
| **L1** | `ols_*` | Rolling OLS: `spread = ALGO − β · equal_basket` → z overlay | Direct upgrade of production `pairs_*` (which hardcodes β≈1) |
| **L2** | `mpairs_*` | Top-k corr-screened instrument pairs, OLS hedge, z overlays | New family; orthogonal to ALGO–basket |

**Isolation rules (same as prior wave):**

- New params are **default-off** so `Params()` still matches production score.
- Track agents never edit `teamName.py` or `eval.py`.
- `build_grid()` is single-owner per tick (no cross-track contamination).
- Lead-only merge into production after midday reconciliation.
- Kill rule: 3 consecutive non-promotable ticks on a track → freeze that track.

**Library policy:**

- Research in `loop.py` may use numpy least-squares (preferred) or
  `statsmodels` / `sklearn` helpers if convenient.
- Anything promoted into `teamName.py` must be **numpy-only**, reimplemented
  to match the harness winner on the same `prices.txt`.

**Docs:**

- `docs/tracks/L1.md`, `docs/tracks/L2.md` — hypothesis banks, reject memory, tick logs
- Updates to `docs/tracks/PROTOCOL.md` and `STRATEGY_LOOP.md` Iteration Log
- Prior tracks A/B/C/D remain historical; not reopened unless a new hypothesis
  family emerges that is absent from their reject memory

---

## Components & signal math

### Shared helpers (harness only)

- `rolling_ols_beta(y, x, lb, intercept=False) -> float` — least-squares β on
  the last `lb` observations (no look-ahead: window ends at current day).
- `spread_z(spread, lb) -> float` — `(last − mean) / (std + VOLATILITY_FLOOR)`.
- Default-off invariance tests: `Params()` ≡ production; `ols_weight=0` and
  `mpairs_weight=0` leave scores unchanged.

### Track L1 — `ols_*`

| Param | Default | Role |
|:---|:---|:---|
| `ols_lookback` | 40 | Window for β and z |
| `ols_weight` | 0.0 | Blend into base signal (0 = off) |
| `ols_entry_z` | 2.0 | Trade only if \|z\| ≥ threshold |
| `ols_intercept` | False | Include intercept in OLS when True |

**Signal:** On ALGO log-price vs equal-weight basket of instruments 1..n,
fit β, form `spread = ALGO − β · basket`, z-score, then same position shape as
current pairs (ALGO vs equal hedge on the rest). Blend:
`(1 − w) · signal + w · pair_sig` when \|z\| ≥ entry.

**Double-overlay rule:** Production already has `pairs_weight=0.20`. L1 grids
must explicitly test:

1. **Replace mode (preferred default for sweeps):** `pairs_weight=0`, weight on `ols_*`
2. **Additive mode:** keep production pairs on and add `ols_*` (expect redundancy)

Do not silently run both ALGO–basket overlays without labeling the mode in the
candidate label / track log.

### Track L2 — `mpairs_*`

| Param | Default | Role |
|:---|:---|:---|
| `mpairs_lookback` | 60 | Screen + OLS + z window |
| `mpairs_weight` | 0.0 | Blend weight (0 = off) |
| `mpairs_top_k` | 5 | Number of pairs traded |
| `mpairs_entry_z` | 2.0 | Per-pair entry threshold |
| `mpairs_min_corr` | 0.85 | Absolute correlation floor for screening |

**Screen:** Among instruments **1..50** (ALGO excluded by default so L2 ≠ L1),
rank pairs by absolute correlation of log returns over `mpairs_lookback`, keep
pairs with `|corr| ≥ mpairs_min_corr`, take top-k.

**Signal:** For each selected pair `(i, j)`, OLS β of log-price i on j, spread
`i − β·j`, z-score; if \|z\| ≥ entry, long/short the pair in hedge ratio space;
average active pair signals; blend into base with `mpairs_weight`.

**Runtime control:** Prefer corr screen every day (cheap). Full Engle-Granger /
ADF cointegration is **out of default path**; may be added later as an optional
Params flag if L2 corr-only fails for a coherent reason.

### Promote path

When `promote: true` for a track winner:

1. Lead confirms midday merge rules (single promote; half2 gate).
2. Port winning constants + numpy OLS into `teamName.py`.
3. Sync `loop.py` `Params` defaults to the new production.
4. Verify: `test_teamName.py` 8/8, `eval.py` score matches harness, numpy-only.
5. Log Iteration entry in `STRATEGY_LOOP.md` and update track doc status.

Optional post-solo tick: if both L1 and L2 are promotable-but-held (or one
promoted and the other strong), one joint `ols_* + mpairs_*` ensemble grid
before a final promote — same pattern as prior Track D.

---

## Protocol, testing, risks

### Execution rhythm

Reuse `docs/tracks/PROTOCOL.md` with tracks L1 and L2:

```
Morning assign → Parallel L1+L2 ticks → Midday merge → Evening log
```

- One hypothesis per track per tick; numbers only from real `loop.py` runs.
- Midday: all reject → no production change; one promote → promote procedure;
  two promote → keep higher official score that still passes
  `half2 ≥ 0.95 × baseline half2`; park the other for ensemble.

### Testing / gates

| Check | Requirement |
|:---|:---|
| Floor | `python eval.py` ≈ **251.70**; `Params()` matches production |
| Default-off | `ols_weight=0`, `mpairs_weight=0` do not change score |
| Promote | Existing `is_promotable` (official ↑ + train/half1/half2 guards) |
| Post-promote | `test_teamName.py` 8/8 + `eval.py` + numpy-only audit |
| Bit-match | Harness winner PnL/score matches `teamName` after port |

### Risks & mitigations

| Risk | Mitigation |
|:---|:---|
| Double ALGO–basket overlay (pairs + ols) | Explicit replace vs additive grids; prefer replace |
| L2 turnover / commissions | High entry_z, small top_k, modest weight; Sharpe-aware score |
| OLS overfit / look-ahead | Rolling windows only on past data inside `strategy_positions` |
| Heavy L2 runtime | Cap grid size; corr screen default (no daily cointegration) |
| Numpy port drift | Promote checklist bit-match on same `prices.txt` |
| Repeating Track A failure mode | L2 is pair-wise OLS, not cross-sectional demean of all names |

### Out of scope

- Reopening rejected Track A xs / momentum / EMA hypotheses
- Editing `eval.py` or `prices.txt`
- Leaderboard submit automation
- Non-numpy dependencies in production `teamName.py`
- L3/L4/L5 from the shortlist (unless L1+L2 both freeze with clear lessons)

### Success criteria

- At least one track completes ≥1 real sweep tick with logged verdict.
- Production score never drops below 251.70 without an explicit lead decision.
- If a promote occurs, official score improves and robustness gates pass.
- If both tracks freeze, STRATEGY_LOOP records clear reject lessons for
  the 16 Jul General Round restart.

---

## Deliverables

1. `loop.py`: shared OLS helpers; `ols_*` and `mpairs_*` Params + signal paths;
   track-owned `build_grid` for research ticks.
2. `docs/tracks/L1.md`, `docs/tracks/L2.md`.
3. PROTOCOL / STRATEGY_LOOP updates for this wave.
4. Promote (only if gate fires): `teamName.py` + Params sync + CYCLING.py refresh.
5. Implementation plan under `docs/superpowers/plans/` (next step after spec approval).
