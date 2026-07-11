# OLS / Multi-Pair Research Tracks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Track L1 (rolling OLS ALGO–basket) and Track L2 (multi-pair OLS) as default-off research knobs in `loop.py`, run first promote-gated sweeps, and promote to `teamName.py` only if the existing gate fires.

**Architecture:** Keep production `teamName.py` frozen at score **251.70**. Extend `loop.py` with shared numpy OLS helpers plus `ols_*` / `mpairs_*` Params. Parallel tracks own `build_grid()` one tick at a time. Lead-only promote with numpy-only port.

**Tech Stack:** Python 3, numpy, pandas, unittest, existing `eval.py` / `loop.py` harness.

**Spec:** `docs/superpowers/specs/2026-07-12-ols-mpairs-tracks-design.md`

---

## File map

| File | Responsibility |
|:---|:---|
| `teamName.py` | Production only — lead edits on promote |
| `loop.py` | Research harness: helpers, Params, signals, grids |
| `eval.py` | Official scorer — **never edit** |
| `prices.txt` | Price data — **never edit** |
| `test_teamName.py` | Production contract tests |
| `test_loop_ols.py` | Unit tests for OLS helpers + default-off invariance |
| `docs/tracks/L1.md` | Track L1 hypothesis bank + logs |
| `docs/tracks/L2.md` | Track L2 hypothesis bank + logs |
| `docs/tracks/PROTOCOL.md` | Add L1/L2 wave rhythm |
| `STRATEGY_LOOP.md` | Iteration log entries |
| `CYCLING.py` | Refresh only if production promotes |

---

### Task 1: Create Track L1 and L2 docs

**Files:**
- Create: `docs/tracks/L1.md`
- Create: `docs/tracks/L2.md`

- [ ] **Step 1: Write `docs/tracks/L1.md`**

```markdown
# Track L1 — Rolling OLS ALGO–basket

**Status:** active  
**Owner:** Agent L1  
**Params prefix:** `ols_*` (default off)  
**Kill rule:** 3 consecutive non-promotable ticks → freeze  
**Floor:** production score 251.70 (`pairs_*` + `algo_signal_scale=3.0`)

## Hypothesis bank
1. Replace equal-weight β≈1 pairs with rolling OLS β on ALGO vs equal basket (replace mode: `pairs_weight=0`).
2. Additive OLS overlay on top of production pairs (expect redundancy — secondary).
3. Intercept-on OLS (`ols_intercept=True`) vs no-intercept.
4. Sensitivity: lookback ∈ {30,40,60} × weight ∈ {0.10,0.20,0.30} × entry_z ∈ {1.5,2.0,2.5}.

## Reject memory
(none yet)

## Log
| Tick | Hypothesis | Best score | train | half2 | Verdict |
|:---|:---|---:|---:|---:|:---|
```

- [ ] **Step 2: Write `docs/tracks/L2.md`**

```markdown
# Track L2 — Multi-pair OLS mean reversion

**Status:** active  
**Owner:** Agent L2  
**Params prefix:** `mpairs_*` (default off)  
**Kill rule:** 3 consecutive non-promotable ticks → freeze  
**Floor:** production score 251.70

## Hypothesis bank
1. Top-k corr-screened pairs among instruments 1..50 (exclude ALGO), OLS hedge, z-entry.
2. Vary top_k ∈ {3,5,8}, min_corr ∈ {0.75,0.85,0.90}, lookback ∈ {40,60,80}.
3. Modest blend weights {0.10,0.20} so L2 does not dominate base reversal.

## Reject memory
(none yet)

## Log
| Tick | Hypothesis | Best score | train | half2 | Verdict |
|:---|:---|---:|---:|---:|:---|
```

- [ ] **Step 3: Commit**

```bash
git add docs/tracks/L1.md docs/tracks/L2.md
git commit -m "docs: add Track L1 and L2 research track stubs"
```

---

### Task 2: Shared OLS helpers + unit tests

**Files:**
- Modify: `loop.py` (add helpers near top of strategy section, before `strategy_positions`)
- Create: `test_loop_ols.py`

- [ ] **Step 1: Write failing tests in `test_loop_ols.py`**

```python
"""Unit tests for loop.py OLS helpers and default-off invariance."""
import unittest

import numpy as np

from loop import (
    Params,
    evaluate,
    load_prices,
    rolling_ols_beta,
    spread_z,
    VOLATILITY_FLOOR,
)


class TestRollingOlsBeta(unittest.TestCase):
    def test_known_slope_no_intercept(self):
        x = np.arange(10, dtype=float)
        y = 2.5 * x
        beta = rolling_ols_beta(y, x, lb=10, intercept=False)
        self.assertAlmostEqual(beta, 2.5, places=6)

    def test_known_slope_with_intercept(self):
        x = np.arange(10, dtype=float)
        y = 3.0 + 1.5 * x
        beta = rolling_ols_beta(y, x, lb=10, intercept=True)
        self.assertAlmostEqual(beta, 1.5, places=5)

    def test_uses_only_last_lb(self):
        x = np.concatenate([np.zeros(5), np.arange(5, dtype=float)])
        y = np.concatenate([np.ones(5) * 100, 2.0 * np.arange(5, dtype=float)])
        beta = rolling_ols_beta(y, x, lb=5, intercept=False)
        self.assertAlmostEqual(beta, 2.0, places=6)


class TestSpreadZ(unittest.TestCase):
    def test_zero_at_mean(self):
        s = np.ones(20)
        self.assertAlmostEqual(spread_z(s, 20), 0.0, places=6)

    def test_positive_when_last_high(self):
        s = np.zeros(20)
        s[-1] = 5.0
        z = spread_z(s, 20)
        self.assertGreater(z, 0.0)


class TestDefaultOffInvariance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prc = load_prices()
        cls.base = evaluate(cls.prc, Params())

    def test_ols_weight_zero_matches_baseline(self):
        cand = evaluate(self.prc, Params(ols_weight=0.0, ols_lookback=40))
        self.assertAlmostEqual(cand.score, self.base.score, places=2)

    def test_mpairs_weight_zero_matches_baseline(self):
        cand = evaluate(self.prc, Params(mpairs_weight=0.0, mpairs_lookback=60))
        self.assertAlmostEqual(cand.score, self.base.score, places=2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests — expect ImportError / missing Params fields**

```bash
python -m unittest test_loop_ols.py -v
```

Expected: FAIL (helpers / Params fields missing).

- [ ] **Step 3: Implement helpers in `loop.py`**

Add after `VOLATILITY_FLOOR` / before or near `_reversal_signal`:

```python
def rolling_ols_beta(
    y: np.ndarray,
    x: np.ndarray,
    lb: int,
    intercept: bool = False,
) -> float:
    """Least-squares slope of y on x using the last lb observations."""
    yy = np.asarray(y[-lb:], dtype=float)
    xx = np.asarray(x[-lb:], dtype=float)
    if intercept:
        X = np.column_stack([np.ones(lb), xx])
        coef, _, _, _ = np.linalg.lstsq(X, yy, rcond=None)
        return float(coef[1])
    denom = float(xx @ xx)
    if denom < VOLATILITY_FLOOR:
        return 0.0
    return float(xx @ yy) / denom


def spread_z(spread: np.ndarray, lb: int) -> float:
    """Z-score of the last point vs the last lb window."""
    window = np.asarray(spread[-lb:], dtype=float)
    mu = float(window.mean())
    sd = float(window.std()) + VOLATILITY_FLOOR
    return (float(window[-1]) - mu) / sd
```

- [ ] **Step 4: Add default-off Params stubs (fields only; signals come in Tasks 3–4)**

In `Params`, after `algo_hedge_weight`:

```python
    # Track L1: rolling OLS ALGO-vs-basket (default off)
    ols_lookback: int = 40
    ols_weight: float = 0.0
    ols_entry_z: float = 2.0
    ols_intercept: bool = False
    # Track L2: multi-pair OLS (default off)
    mpairs_lookback: int = 60
    mpairs_weight: float = 0.0
    mpairs_top_k: int = 5
    mpairs_entry_z: float = 2.0
    mpairs_min_corr: float = 0.85
```

Extend `label()` similarly:

```python
        ols = (
            f" ols={self.ols_lookback}@{self.ols_weight:.2f}"
            f"z{self.ols_entry_z:.2f}{'+int' if self.ols_intercept else ''}"
            if self.ols_weight > 0
            else ""
        )
        mpairs = (
            f" mpairs={self.mpairs_lookback}@{self.mpairs_weight:.2f}"
            f"k{self.mpairs_top_k}z{self.mpairs_entry_z:.2f}"
            f"c{self.mpairs_min_corr:.2f}"
            if self.mpairs_weight > 0
            else ""
        )
```

Append `{ols}{mpairs}` to the returned label string.

- [ ] **Step 5: Re-run unit tests**

```bash
python -m unittest test_loop_ols.py -v
```

Expected: helper tests PASS. Invariance tests PASS (weights 0 ⇒ no signal path yet, or after Tasks 3–4 still pass). Baseline:

```bash
python loop.py 2>&1 | tail -8
```

Expected: Score **251.70**.

- [ ] **Step 6: Commit**

```bash
git add loop.py test_loop_ols.py
git commit -m "feat: add rolling OLS helpers and L1/L2 Params stubs"
```

---

### Task 3: Track L1 signal path + first research tick

**Files:**
- Modify: `loop.py` (`strategy_positions`, `build_grid`)
- Modify: `docs/tracks/L1.md`
- Modify: `STRATEGY_LOOP.md`

- [ ] **Step 1: Wire OLS overlay in `strategy_positions`**

After the existing `pairs_weight` block (and before `algo_signal_scale`), add:

```python
    if params.ols_weight > 0:
        need = max(need, params.ols_lookback)
    # (also extend the top `longest` / early `need` calculations the same way)

    if params.ols_weight > 0 and nt > params.ols_lookback:
        lb = params.ols_lookback
        basket = np.nanmean(log_prices[1:, :], axis=0)
        algo = log_prices[0, :]
        beta = rolling_ols_beta(algo, basket, lb, intercept=params.ols_intercept)
        spread = algo - beta * basket
        z = spread_z(spread, lb)
        if abs(z) >= params.ols_entry_z:
            ols_sig = np.zeros(nins)
            ols_sig[0] = -np.clip(z, -params.signal_clip, params.signal_clip)
            ols_sig[1:] = -ols_sig[0] / (nins - 1)
            signal = (1.0 - params.ols_weight) * signal + params.ols_weight * ols_sig
```

Also extend the early `longest` / `need` blocks:

```python
    if params.ols_weight > 0:
        longest = max(longest, params.ols_lookback)
    # ...
    if params.ols_weight > 0:
        need = max(need, params.ols_lookback)
```

- [ ] **Step 2: Set `build_grid()` to Track L1 only (replace + additive)**

Replace `build_grid` body with:

```python
def build_grid() -> list[Params]:
    """Track L1: rolling OLS ALGO-basket. Prefer replace mode (pairs_weight=0)."""
    grid = [Params()]  # production floor
    # Replace mode: turn off β≈1 pairs, put weight on ols_*
    for lb in (30, 40, 60):
        for w in (0.10, 0.20, 0.30):
            for z in (1.5, 2.0, 2.5):
                grid.append(Params(
                    pairs_weight=0.0,
                    ols_lookback=lb,
                    ols_weight=w,
                    ols_entry_z=z,
                    ols_intercept=False,
                ))
    # Additive smoke: keep production pairs, add one OLS candidate
    grid.append(Params(
        ols_lookback=40, ols_weight=0.10, ols_entry_z=2.0, ols_intercept=False,
    ))
    # Intercept variant (replace)
    grid.append(Params(
        pairs_weight=0.0,
        ols_lookback=40, ols_weight=0.20, ols_entry_z=2.0, ols_intercept=True,
    ))
    return grid
```

- [ ] **Step 3: Default-off invariance + sweep**

```bash
python -m unittest test_loop_ols.py -v
python loop.py 2>&1 | tail -8
python loop.py --sweep --csv results.csv 2>&1 | tee /tmp/l1_sweep.txt
python loop.py --json 2>&1 | tee /tmp/l1_json.txt
```

Expected: baseline still 251.70; JSON reports `promote: true/false` with real numbers.

- [ ] **Step 4: Log results**

Append a row to `docs/tracks/L1.md` Log table and an Iteration entry to `STRATEGY_LOOP.md` with hypothesis, best score, train, half2, verdict. Do **not** edit `teamName.py`.

- [ ] **Step 5: Commit**

```bash
git add loop.py docs/tracks/L1.md STRATEGY_LOOP.md results.csv
git commit -m "research: Track L1 OLS ALGO-basket first sweep"
```

(Omit `results.csv` from the commit if the repo conventionally leaves it untracked.)

---

### Task 4: Track L2 signal path + first research tick

**Files:**
- Modify: `loop.py` (`strategy_positions`, `build_grid`)
- Modify: `docs/tracks/L2.md`
- Modify: `STRATEGY_LOOP.md`

- [ ] **Step 1: Add multi-pair helper in `loop.py`**

```python
def _mpairs_signal(
    log_prices: np.ndarray,
    daily_returns: np.ndarray,
    params: Params,
) -> np.ndarray:
    """Corr-screened top-k pair OLS residual signals (instruments 1..n-1)."""
    nins = log_prices.shape[0]
    lb = params.mpairs_lookback
    sig = np.zeros(nins)
    # Use log returns over lookback for correlation screen
    rets = daily_returns[1:, -lb:]  # exclude ALGO (index 0)
    n = rets.shape[0]
    if n < 2 or lb < 3:
        return sig
    # Pairwise abs corr
    candidates: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = rets[i], rets[j]
            if a.std() < VOLATILITY_FLOOR or b.std() < VOLATILITY_FLOOR:
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            if abs(corr) >= params.mpairs_min_corr:
                candidates.append((abs(corr), i + 1, j + 1))  # map back to instrument idx
    candidates.sort(reverse=True)
    chosen = candidates[: params.mpairs_top_k]
    if not chosen:
        return sig
    active = 0
    for _, i, j in chosen:
        yi = log_prices[i, :]
        xj = log_prices[j, :]
        beta = rolling_ols_beta(yi, xj, lb, intercept=False)
        spread = yi - beta * xj
        z = spread_z(spread, lb)
        if abs(z) < params.mpairs_entry_z:
            continue
        zc = float(np.clip(z, -params.signal_clip, params.signal_clip))
        # Mean-revert: short spread if z>0 → short i, long j * beta (sign via dollars later)
        sig[i] += -zc
        sig[j] += zc * beta
        active += 1
    if active == 0:
        return np.zeros(nins)
    sig /= active
    # Renormalise so max abs <= signal_clip
    m = np.max(np.abs(sig)) + VOLATILITY_FLOOR
    return np.clip(sig / m, -params.signal_clip, params.signal_clip) * params.signal_clip
```

- [ ] **Step 2: Wire into `strategy_positions`**

Extend `longest` / `need` for `mpairs_weight > 0` using `mpairs_lookback`. After OLS block (before algo scale):

```python
    if params.mpairs_weight > 0 and nt > params.mpairs_lookback:
        mp = _mpairs_signal(log_prices, daily_returns, params)
        signal = (1.0 - params.mpairs_weight) * signal + params.mpairs_weight * mp
```

- [ ] **Step 3: Set `build_grid()` to Track L2 only**

```python
def build_grid() -> list[Params]:
    """Track L2: multi-pair OLS. Production pairs/scale left at defaults."""
    grid = [Params()]
    for lb in (40, 60):
        for k in (3, 5):
            for w in (0.10, 0.20):
                for z in (2.0, 2.5):
                    for cmin in (0.85, 0.90):
                        grid.append(Params(
                            mpairs_lookback=lb,
                            mpairs_weight=w,
                            mpairs_top_k=k,
                            mpairs_entry_z=z,
                            mpairs_min_corr=cmin,
                        ))
    return grid
```

Keep grid ≤ ~35 candidates so runtime stays reasonable.

- [ ] **Step 4: Test + sweep**

```bash
python -m unittest test_loop_ols.py -v
python loop.py 2>&1 | tail -8
python loop.py --sweep --csv results.csv 2>&1 | tee /tmp/l2_sweep.txt
python loop.py --json 2>&1 | tee /tmp/l2_json.txt
```

Expected: baseline 251.70; L2 candidates scored; JSON verdict logged.

- [ ] **Step 5: Log to `docs/tracks/L2.md` and `STRATEGY_LOOP.md`. Do not edit `teamName.py`.**

- [ ] **Step 6: Commit**

```bash
git add loop.py docs/tracks/L2.md STRATEGY_LOOP.md
git commit -m "research: Track L2 multi-pair OLS first sweep"
```

---

### Task 5: PROTOCOL update for L1/L2 wave

**Files:**
- Modify: `docs/tracks/PROTOCOL.md`

- [ ] **Step 1: Append an L1/L2 wave section**

Add at the end of `PROTOCOL.md`:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add docs/tracks/PROTOCOL.md
git commit -m "docs: add Wave 2 L1/L2 protocol notes"
```

---

### Task 6: Midday merge / promote-or-hold

**Files:**
- Possibly modify: `teamName.py`, `loop.py` Params defaults, `CYCLING.py`, track docs, `STRATEGY_LOOP.md`

- [ ] **Step 1: Collect verdicts**

From Task 3 and Task 4 JSON outputs, record:

| Track | Best official | half2 | promote? |
|:---|---:|---:|:---|
| L1 | … | … | true/false |
| L2 | … | … | true/false |

- [ ] **Step 2: Decide**

- Both false → no production change; append merge note to STRATEGY_LOOP; skip Steps 3–5.
- Exactly one true → proceed to promote that winner (Steps 3–5).
- Both true → keep higher official score that still has `half2 ≥ 0.95 × baseline half2` (baseline half2 ≈ 167.60); park the other for Task 7.

- [ ] **Step 3: Promote (only if Step 2 says promote)**

Port winning constants + numpy OLS into `teamName.py`. Sync `Params` defaults in `loop.py`. Copy to `CYCLING.py`.

- [ ] **Step 4: Verify promote**

```bash
python -m unittest test_teamName.py test_loop_ols.py -v
python eval.py 2>&1 | tail -5
python loop.py 2>&1 | tail -8
```

Expected: tests pass; eval score equals new harness baseline; `loop.py --json` now `promote: false` (winner is floor).

- [ ] **Step 5: Log + commit**

```bash
git add teamName.py loop.py CYCLING.py docs/tracks/L1.md docs/tracks/L2.md STRATEGY_LOOP.md
git commit -m "feat: promote L1/L2 winner to production"
```

Or if no promote:

```bash
git add docs/tracks/L1.md docs/tracks/L2.md STRATEGY_LOOP.md
git commit -m "docs: Wave 2 midday merge — no promote"
```

---

### Task 7: Optional L1×L2 ensemble tick (only if both tracks had strong candidates)

**Skip entirely if Task 6 already promoted a clear winner and the other track was weak (best score < baseline), or if both were hard rejects.**

**Files:**
- Modify: `loop.py` `build_grid`
- Modify: track docs / STRATEGY_LOOP

- [ ] **Step 1: Build joint grid** from each track’s best 1–2 param sets (replace-mode L1 + L2 weights), include `Params()` baseline.

- [ ] **Step 2: Sweep + JSON**

```bash
python loop.py --sweep 2>&1 | tee /tmp/l12_ens.txt
python loop.py --json 2>&1 | tee /tmp/l12_ens_json.txt
```

- [ ] **Step 3: If promote true, run Task 6 Steps 3–5 for the ensemble winner; else log reject and commit docs only.**

---

## Self-review checklist (plan author)

| Spec requirement | Task |
|:---|:---|
| Shared `rolling_ols_beta` / `spread_z` | Task 2 |
| `ols_*` Params + signal + replace/additive grids | Task 3 |
| `mpairs_*` Params + corr screen + OLS pairs | Task 4 |
| Track docs L1/L2 | Task 1 |
| PROTOCOL update | Task 5 |
| Lead-only promote + numpy port | Task 6 |
| Optional ensemble | Task 7 |
| Default-off invariance | Tasks 2–4 |
| Never edit eval.py / prices.txt | All tasks |

No TBD/TODO placeholders. Param names consistent: `ols_*`, `mpairs_*`.
