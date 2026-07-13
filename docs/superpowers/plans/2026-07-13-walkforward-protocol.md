# Walk-Forward Protocol Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `loop.py` promote gates to expanding-window walk-forward (F1–F3 majority + official ↑ + F3 gate), reset the research floor to a minimal core, and stand up parallel tracks R1 (OLS) and R2 (multi-pair) without touching live `teamName.py` until a lead promote.

**Architecture:** Generalize simulation to score arbitrary day ranges; evaluate every candidate on three expanding OOS folds plus the official last-250 window; replace half1/half2/train promote logic with majority-of-folds + F3 guard. Keep signal code (`ols_*`, `mpairs_*`) as-is behind default-off weights. Live production file stays frozen until gates fire.

**Tech Stack:** Python 3, numpy, pandas, unittest, existing `eval.py` / `loop.py` harness.

**Spec:** `docs/superpowers/specs/2026-07-13-walkforward-protocol-design.md`

---

## File map

| File | Responsibility |
|:---|:---|
| `loop.py` | Harness: `simulate_range`, walk-forward `evaluate` / `is_promotable`, minimal-core `Params()`, R1/R2 grids |
| `eval.py` | Official scorer — **never edit** |
| `prices.txt` | Price data — **never edit** |
| `teamName.py` | Live production — **frozen** until lead promote (Task 8 only) |
| `test_loop_ols.py` | OLS helpers + floor invariance + walk-forward gate tests |
| `test_teamName.py` | Production contract — unchanged until promote |
| `docs/tracks/R1.md` | OLS track bank + logs |
| `docs/tracks/R2.md` | Multi-pair track bank + logs |
| `docs/tracks/PROTOCOL.md` | Replace wave rhythm with R1/R2 walk-forward protocol |
| `AGENT_LOOP.md` | Align agent iteration steps to new gates |
| `STRATEGY_LOOP.md` | Wave kickoff + tick logs |
| `CYCLING.py` | Refresh only on lead promote |

---

### Task 1: Create Track R1 and R2 docs

**Files:**
- Create: `docs/tracks/R1.md`
- Create: `docs/tracks/R2.md`

- [ ] **Step 1: Write `docs/tracks/R1.md`**

```markdown
# Track R1 — Rolling OLS ALGO–basket

**Status:** active  
**Owner:** Agent R1  
**Params prefix:** `ols_*` (default off)  
**Kill rule:** 3 consecutive non-promotable ticks → freeze  
**Floor:** minimal core (reversal 5/20 + band 0.195 + regime 10/60@1.15→0.22; all overlays off)  
**Gates:** walk-forward F1–F3 majority + official ↑ + F3 ≥ 0.95×base F3

## Hypothesis bank
1. Rolling OLS β on ALGO vs equal basket; z-entry mean-reversion overlay on minimal core.
2. Lookback ∈ {30,40,60} × weight ∈ {0.10,0.20,0.30} × entry_z ∈ {1.5,2.0,2.5}; intercept off.
3. Intercept-on OLS (`ols_intercept=True`) as a secondary arm after tick-1.

## Reject memory
(none yet — prior L1 promote was under old gates / old floor; re-discover)

## Log
| Tick | Hypothesis | Official | F1 | F2 | F3 | Verdict |
|:---|:---|---:|---:|---:|---:|:---|
```

- [ ] **Step 2: Write `docs/tracks/R2.md`**

```markdown
# Track R2 — Multi-pair OLS mean reversion

**Status:** active  
**Owner:** Agent R2  
**Params prefix:** `mpairs_*` (default off)  
**Kill rule:** 3 consecutive non-promotable ticks → freeze  
**Floor:** minimal core (same as R1)  
**Gates:** walk-forward F1–F3 majority + official ↑ + F3 ≥ 0.95×base F3

## Hypothesis bank
1. Top-k corr-screened pairs among instruments 1..50 (exclude ALGO), OLS hedge, z-entry.
2. Lookback ∈ {40,60} × top_k ∈ {3,5} × weight ∈ {0.10,0.20} × entry_z ∈ {1.5,2.0} × min_corr ∈ {0.55,0.65,0.70}.
3. Confirm active-pair counts before expanding grids (prior L2 failed with min_corr above empirical ~0.77).

## Reject memory
- Legacy L2 (old floor/gates): min_corr 0.85/0.90 selected zero pairs; ungated blend diluted baseline.

## Log
| Tick | Hypothesis | Official | F1 | F2 | F3 | Verdict |
|:---|:---|---:|---:|---:|---:|:---|
```

- [ ] **Step 3: Commit**

```bash
git add docs/tracks/R1.md docs/tracks/R2.md
git commit -m "docs: add Track R1 and R2 walk-forward research stubs"
```

---

### Task 2: Add `simulate_range` + failing tests for fold windows

**Files:**
- Modify: `loop.py` (near `simulate`)
- Modify: `test_loop_ols.py`

- [ ] **Step 1: Write failing tests for range simulation**

Add to `test_loop_ols.py`:

```python
from loop import Params, evaluate, load_prices, simulate, simulate_range, is_promotable


class TestSimulateRange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prc = load_prices()
        cls.nt = cls.prc.shape[1]

    def test_official_range_matches_simulate(self):
        p = Params()
        a = simulate(self.prc, p, 250)
        # last 250 days: start_day = nt-250, end_day = nt
        b = simulate_range(self.prc, p, start_day=self.nt - 250, end_day=self.nt)
        self.assertAlmostEqual(a.score, b.score, places=4)
        self.assertEqual(len(a.pll), len(b.pll))
        np.testing.assert_allclose(a.pll, b.pll, rtol=1e-9, atol=1e-9)

    def test_fold_length_is_100(self):
        p = Params()
        # F1: days 201-300 → start_day=200, end_day=300
        r = simulate_range(self.prc, p, start_day=200, end_day=300)
        self.assertEqual(len(r.pll), 100)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest test_loop_ols.TestSimulateRange -v`

Expected: FAIL with `ImportError` / `simulate_range` not defined.

- [ ] **Step 3: Implement `simulate_range` and refactor `simulate`**

In `loop.py`, replace `simulate` body extraction with:

```python
def simulate_range(
    prc_all: np.ndarray,
    params: Params,
    start_day: int,
    end_day: int,
) -> Result:
    """Simulate positions from start_day..end_day; score PnL for t > start_day.

    Day indices match the existing simulate() convention: for a series of length
    nt, official last-250 uses start_day=nt-250, end_day=nt.
    """
    nins, nt = prc_all.shape
    if not (0 <= start_day < end_day <= nt):
        raise ValueError(f"bad range: start_day={start_day} end_day={end_day} nt={nt}")

    comm_rate = np.full(nins, DEFAULT_COMM_RATE)
    comm_rate[0] = INST0_COMM_RATE
    dlr_limit = np.full(nins, float(params.default_dollar_limit))
    dlr_limit[0] = float(params.algo_dollar_limit)

    cash = 0.0
    cur_pos = np.zeros(nins)
    tot_dvolume = 0.0
    value = 0.0
    comm = 0.0
    pll: list[float] = []

    for t in range(start_day, end_day + 1):
        prc_so_far = prc_all[:, :t]
        cur_prices = prc_so_far[:, -1]

        if t < end_day:
            new_orig = strategy_positions(prc_so_far, cur_pos.astype(int), params)
            pos_limits = (dlr_limit / cur_prices).astype(int)
            new_pos = np.clip(new_orig, -pos_limits, pos_limits).astype(int)
        else:
            new_pos = np.array(cur_pos)

        delta = new_pos - cur_pos
        cash -= cur_prices.dot(delta) + comm
        dvolumes = cur_prices * np.abs(delta)
        tot_dvolume += float(np.sum(dvolumes))
        comm = float(np.sum(dvolumes * comm_rate))

        cur_pos = np.array(new_pos)
        pos_value = float(cur_pos.dot(cur_prices))
        today_pl = cash + pos_value - value
        value = cash + pos_value

        if t > start_day:
            pll.append(today_pl)

    pll_arr = np.array(pll)
    mu, std = float(np.mean(pll_arr)), float(np.std(pll_arr))
    sharpe = np.sqrt(250) * mu / std if std > 0 else 0.0
    return Result(mu, std, sharpe, _score(mu, std), tot_dvolume, pll_arr)


def simulate(prc_all: np.ndarray, params: Params, num_test_days: int) -> Result:
    """Day-by-day simulation identical in mechanics to eval.py's calcPL."""
    nt = prc_all.shape[1]
    return simulate_range(prc_all, params, start_day=nt - num_test_days, end_day=nt)
```

Note: if current `simulate` hardcodes `DEFAULT_DLR_LIMIT` / `INST0_DLR_LIMIT` instead of `params.*`, keep the same constants the file already uses so official bit-match is preserved — prefer matching existing `simulate` dollar-limit source exactly when refactoring.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest test_loop_ols.TestSimulateRange -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add loop.py test_loop_ols.py
git commit -m "feat: add simulate_range for walk-forward fold windows"
```

---

### Task 3: Walk-forward `Evaluation` + `is_promotable` (TDD)

**Files:**
- Modify: `loop.py` (`Evaluation`, `evaluate`, `is_promotable`, CSV/print/`verdict_json`)
- Modify: `test_loop_ols.py`

Fold constants (add near top of `loop.py`):

```python
# Expanding walk-forward folds (score windows on 500-day series)
# F1: days 201-300, F2: 301-400, F3: 401-500  (1-based day labels in the spec)
WALK_FOLDS = (
    (200, 300),  # F1
    (300, 400),  # F2
    (400, 500),  # F3 — end_day must equal nt when nt==500
)
```

If `nt != 500`, derive folds from `nt` in `evaluate` as:
`[(nt - 300, nt - 200), (nt - 200, nt - 100), (nt - 100, nt)]` so the plan stays correct on the current file.

- [ ] **Step 1: Write failing gate tests**

Add to `test_loop_ols.py`:

```python
class TestWalkForwardPromote(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prc = load_prices()

    def test_evaluate_exposes_three_folds(self):
        e = evaluate(self.prc, Params())
        self.assertTrue(hasattr(e, "fold1_score"))
        self.assertTrue(hasattr(e, "fold2_score"))
        self.assertTrue(hasattr(e, "fold3_score"))
        self.assertIsInstance(e.robust, bool)

    def test_baseline_not_promotable_against_itself(self):
        base = evaluate(self.prc, Params())
        self.assertFalse(is_promotable(base, base))

    def test_promotable_requires_majority_and_official_and_f3(self):
        base = evaluate(self.prc, Params())
        # Synthetic candidate: copy base then bump fields
        from dataclasses import replace

        # Majority (2/3) + official up + F3 ok → promote
        good = replace(
            base,
            score=base.score + 1.0,
            fold1_score=base.fold1_score + 1.0,
            fold2_score=base.fold2_score + 1.0,
            fold3_score=base.fold3_score,  # == base → passes 0.95 gate
            robust=True,
        )
        # Force robust True with fake fold wins: evaluate sets robust; here we set manually
        good = replace(
            good,
            robust=True,
        )
        self.assertTrue(is_promotable(base, good))

        # Official up but only 1/3 folds beat base → not robust / not promotable
        weak = replace(
            base,
            score=base.score + 5.0,
            fold1_score=base.fold1_score + 1.0,
            fold2_score=base.fold2_score - 1.0,
            fold3_score=base.fold3_score - 1.0,
            robust=False,
        )
        self.assertFalse(is_promotable(base, weak))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest test_loop_ols.TestWalkForwardPromote -v`

Expected: FAIL (`fold1_score` missing / old `is_promotable` behavior).

- [ ] **Step 3: Rewrite `Evaluation`, `evaluate`, `is_promotable`**

```python
@dataclass
class Evaluation:
    params: Params
    score: float          # official (last NUM_TEST_DAYS)
    mean_pl: float
    sharpe: float
    dvol: float
    fold1_score: float
    fold2_score: float
    fold3_score: float
    robust: bool

    def row(self) -> dict:
        d = asdict(self.params)
        d.update(
            label=self.params.label(),
            score=round(self.score, 2),
            mean_pl=round(self.mean_pl, 2),
            sharpe=round(self.sharpe, 3),
            dvol=round(self.dvol, 0),
            fold1=round(self.fold1_score, 2),
            fold2=round(self.fold2_score, 2),
            fold3=round(self.fold3_score, 2),
            robust=self.robust,
        )
        return d


def _fold_bounds(nt: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Expanding 100-day OOS folds ending at nt, nt-100, nt-200."""
    return (
        (nt - 300, nt - 200),
        (nt - 200, nt - 100),
        (nt - 100, nt),
    )


def evaluate(prc_all: np.ndarray, params: Params) -> Evaluation:
    """Official score + expanding walk-forward fold scores."""
    nt = prc_all.shape[1]
    official = simulate(prc_all, params, NUM_TEST_DAYS)
    f1, f2, f3 = _fold_bounds(nt)
    r1 = simulate_range(prc_all, params, *f1)
    r2 = simulate_range(prc_all, params, *f2)
    r3 = simulate_range(prc_all, params, *f3)

    # Robustness needs the floor comparison; evaluate() alone cannot know base.
    # Store raw fold positivity here; majority-vs-base is applied in is_promotable
    # via a helper that sets cand.robust when comparing — see below.
    fold_positive = r1.score > 0 and r2.score > 0 and r3.score > 0
    return Evaluation(
        params=params,
        score=official.score,
        mean_pl=official.mean_pl,
        sharpe=official.sharpe,
        dvol=official.dvol,
        fold1_score=r1.score,
        fold2_score=r2.score,
        fold3_score=r3.score,
        robust=fold_positive,  # provisional; finalize in mark_robust/is_promotable
    )


def mark_robust(base: Evaluation, cand: Evaluation) -> bool:
    """Spec robust: all folds > 0 AND beat base on ≥ 2 of 3 folds."""
    if not (cand.fold1_score > 0 and cand.fold2_score > 0 and cand.fold3_score > 0):
        return False
    wins = sum(
        [
            cand.fold1_score > base.fold1_score,
            cand.fold2_score > base.fold2_score,
            cand.fold3_score > base.fold3_score,
        ]
    )
    return wins >= 2


def is_promotable(base: Evaluation, cand: Evaluation) -> bool:
    """Majority walk-forward + official ↑ + F3 ≥ 0.95 × base F3."""
    return (
        mark_robust(base, cand)
        and cand.score > base.score
        and cand.fold3_score >= base.fold3_score * 0.95
    )
```

Update `_result_fieldnames` eval keys to:
`("label", "score", "mean_pl", "sharpe", "dvol", "fold1", "fold2", "fold3", "robust")`.

Update `run_sweep` / `main` printouts to show `fold1/fold2/fold3` instead of `half1/half2/train`.

In `run_sweep` / `verdict_json`, when printing robust flag use `mark_robust(base_eval, e)` (not provisional `e.robust` alone). Optionally set `e.robust = mark_robust(base_eval, e)` after base is known before printing/logging.

- [ ] **Step 4: Run gate tests**

Run: `python -m unittest test_loop_ols.TestWalkForwardPromote -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add loop.py test_loop_ols.py
git commit -m "feat: replace half-split gates with expanding walk-forward promote"
```

---

### Task 4: Reset research `Params()` to minimal core + update invariance tests

**Files:**
- Modify: `loop.py` (`Params` defaults)
- Modify: `test_loop_ols.py`

- [ ] **Step 1: Change `Params` defaults to minimal core**

In `loop.py` `Params`:

```python
algo_signal_scale: float = 1.0   # was 3.0 — research floor turns ALGO scale off
ols_lookback: int = 30
ols_weight: float = 0.0          # was 0.20 — OLS off at research floor
ols_entry_z: float = 1.5
ols_intercept: bool = False
pairs_weight: float = 0.0        # already 0
mpairs_weight: float = 0.0       # already 0
# keep lookbacks=(5,20), rebalance_band=0.195, regime 10/60@1.15→0.22
```

Do **not** edit `teamName.py` in this task.

- [ ] **Step 2: Update `test_loop_ols.py` baseline expectation**

Replace production score lock with a dynamic floor lock:

```python
class TestMinimalCoreFloor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prc = load_prices()
        cls.base = evaluate(cls.prc, Params())

    def test_overlays_default_off(self):
        p = Params()
        self.assertEqual(p.ols_weight, 0.0)
        self.assertEqual(p.mpairs_weight, 0.0)
        self.assertEqual(p.pairs_weight, 0.0)
        self.assertEqual(p.algo_signal_scale, 1.0)
        self.assertEqual(p.xs_weight, 0.0)
        self.assertEqual(p.momentum_weight, 0.0)

    def test_zero_ols_lookback_change_is_noop(self):
        a = evaluate(self.prc, Params(ols_weight=0.0))
        b = evaluate(self.prc, Params(ols_weight=0.0, ols_lookback=60))
        self.assertAlmostEqual(a.score, b.score, places=2)

    def test_zero_mpairs_is_noop(self):
        a = evaluate(self.prc, Params())
        b = evaluate(self.prc, Params(mpairs_weight=0.0, mpairs_lookback=80))
        self.assertAlmostEqual(a.score, b.score, places=2)

    def test_inactive_mpairs_does_not_dilute(self):
        base = evaluate(self.prc, Params())
        cand = evaluate(
            self.prc,
            Params(mpairs_weight=0.10, mpairs_min_corr=0.85, mpairs_lookback=40),
        )
        self.assertAlmostEqual(cand.score, base.score, places=2)

    def test_floor_score_is_positive(self):
        # Record exact value in STRATEGY_LOOP after first run; assert sanity here.
        self.assertGreater(self.base.score, 0.0)
```

Remove `test_production_baseline_score` that asserts `265.18`.

- [ ] **Step 3: Record the new floor score**

Run:

```bash
python loop.py
python -m unittest test_loop_ols.py -v
```

Expected: tests PASS; `loop.py` prints minimal-core official score (note the number for Task 5 docs). Confirm live production is unchanged:

```bash
python eval.py
```

Expected: still **265.18** from current `teamName.py` (research floor ≠ live file).

- [ ] **Step 4: Commit**

```bash
git add loop.py test_loop_ols.py
git commit -m "refactor: reset loop Params research floor to minimal core"
```

---

### Task 5: Rewrite protocol + agent-loop docs for walk-forward wave

**Files:**
- Modify: `docs/tracks/PROTOCOL.md` (prepend / replace wave section for R1/R2)
- Modify: `AGENT_LOOP.md` (gates + promote steps)
- Modify: `STRATEGY_LOOP.md` (wave kickoff iteration entry)

- [ ] **Step 1: Update `AGENT_LOOP.md` objective + decide steps**

Replace the baseline score blurb and step-4 promote description so they match the spec:

- Research baseline = minimal-core `Params()` (not live 265.18).
- Verdict from `python loop.py --json` uses walk-forward `promote`.
- Reject if majority folds fail, official does not rise, or F3 gate fails.
- Do not promote on official score alone.
- Keep hard guardrails (never edit `eval.py` / `prices.txt`; lead-only `teamName.py`).

- [ ] **Step 2: Update `docs/tracks/PROTOCOL.md`**

Add a clear **Wave: Walk-forward R1/R2** section at the top (or replace Day 1–2 A/B/C as historical):

- Morning assign / parallel R1+R2 / midday merge / evening log
- Kill rule (3 rejects)
- Lead-only promote
- Floor check: `python loop.py` minimal-core score (record number from Task 4)
- Note: live `eval.py` on `teamName.py` may still show 265.18 until promote

- [ ] **Step 3: Append STRATEGY_LOOP kickoff entry**

```markdown
### Iteration 34 (Walk-forward protocol kickoff)

- **Date:** 2026-07-13
- **Hypothesis:** Expanding-window majority gates + minimal-core floor will re-discover OLS/pairs without overfitting the official window.
- **Strategy:** Research harness only — Params() reset to minimal core; live teamName.py frozen at 265.18 until lead promote.
- **Result:** Research floor official score = <PASTE FROM Task 4>; gates = F1–F3 majority + official ↑ + F3≥0.95×base.
- **Decision:** KEEP & HARDEN (protocol); begin Track R1/R2 ticks.
- **Learnings:** Research floor ≠ live production until promote.
```

- [ ] **Step 4: Commit**

```bash
git add AGENT_LOOP.md docs/tracks/PROTOCOL.md STRATEGY_LOOP.md
git commit -m "docs: adopt walk-forward R1/R2 protocol and agent loop"
```

---

### Task 6: Track R1 tick 1 — OLS grid under new gates

**Files:**
- Modify: `loop.py` (`build_grid` only)
- Modify: `docs/tracks/R1.md`
- Modify: `STRATEGY_LOOP.md`

- [ ] **Step 1: Set `build_grid()` to R1 tick-1 grid**

```python
def build_grid() -> list[Params]:
    """Track R1 tick 1: OLS ALGO-basket on minimal core."""
    grid = [Params()]  # floor
    for lb, w, z in itertools.product((30, 40, 60), (0.10, 0.20, 0.30), (1.5, 2.0, 2.5)):
        grid.append(
            Params(
                ols_lookback=lb,
                ols_weight=w,
                ols_entry_z=z,
                ols_intercept=False,
            )
        )
    return grid
```

- [ ] **Step 2: Run sweep + JSON verdict**

```bash
python loop.py --sweep
python loop.py --json | head -c 4000
```

Record: baseline official/F1/F2/F3; best candidate; `promote` true/false.

- [ ] **Step 3: Log results (do not edit `teamName.py`)**

Update `docs/tracks/R1.md` log row and append STRATEGY_LOOP Iteration entry with real numbers and `HOLD` / `promote false` as appropriate.

- [ ] **Step 4: Reset `build_grid` to floor-only**

```python
def build_grid() -> list[Params]:
    return [Params()]
```

- [ ] **Step 5: Commit**

```bash
git add loop.py docs/tracks/R1.md STRATEGY_LOOP.md
git commit -m "research: Track R1 OLS first walk-forward sweep"
```

---

### Task 7: Track R2 tick 1 — multi-pair grid under new gates

**Files:**
- Modify: `loop.py` (`build_grid` only)
- Modify: `docs/tracks/R2.md`
- Modify: `STRATEGY_LOOP.md`

- [ ] **Step 1: Set `build_grid()` to R2 tick-1 grid**

```python
def build_grid() -> list[Params]:
    """Track R2 tick 1: corr-screened multi-pair OLS on minimal core."""
    grid = [Params()]
    for lb, k, w, z, cmin in itertools.product(
        (40, 60),
        (3, 5),
        (0.10, 0.20),
        (1.5, 2.0),
        (0.55, 0.65, 0.70),
    ):
        grid.append(
            Params(
                mpairs_lookback=lb,
                mpairs_top_k=k,
                mpairs_weight=w,
                mpairs_entry_z=z,
                mpairs_min_corr=cmin,
            )
        )
    return grid
```

- [ ] **Step 2: Run sweep + JSON**

```bash
python loop.py --sweep
python loop.py --json | head -c 4000
```

Before trusting rejects: spot-check that at least one candidate with `min_corr=0.55` has non-zero overlay activity (score ≠ floor). If all weight>0 candidates tie the floor, stop and fix screening — do not log a false “no edge” reject.

- [ ] **Step 3: Log + reset `build_grid` to `[Params()]`**

Same pattern as Task 6. Never edit `teamName.py`.

- [ ] **Step 4: Commit**

```bash
git add loop.py docs/tracks/R2.md STRATEGY_LOOP.md
git commit -m "research: Track R2 multi-pair first walk-forward sweep"
```

---

### Task 8: Lead midday merge / promote (only if a gate fires)

**Files:**
- Modify only if `promote: true` from Task 6 and/or 7: `teamName.py`, `loop.py` Params defaults, `CYCLING.py`, track docs, `STRATEGY_LOOP.md`

- [ ] **Step 1: Reconcile**

| Case | Action |
|:---|:---|
| Both false | No code promote; log “no ship” in STRATEGY_LOOP |
| One true | Promote that winner |
| Both true | Higher official that still passes F3 ≥ 0.95×base F3; park other for later ensemble |

- [ ] **Step 2: If promoting — port to `teamName.py`**

Copy winning overlay logic/constants into `getMyPosition` (numpy-only). Sync `Params()` defaults in `loop.py` to the winner so research floor becomes the new production. Copy `teamName.py` → `CYCLING.py`.

- [ ] **Step 3: Verify**

```bash
python eval.py
python loop.py
python -m unittest test_teamName.py test_loop_ols.py -v
```

Expected: `eval.py` official score matches harness winner; tests green; `loop.py --json` with floor-only grid shows `promote: false` (baseline == winner).

- [ ] **Step 4: Commit**

```bash
git add teamName.py CYCLING.py loop.py docs/tracks/R1.md docs/tracks/R2.md STRATEGY_LOOP.md
git commit -m "feat: promote walk-forward winner to production"
```

If nothing promoted:

```bash
git add STRATEGY_LOOP.md docs/tracks/R1.md docs/tracks/R2.md
git commit -m "docs: log walk-forward R1/R2 tick-1 with no promote"
```

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|:---|:---|
| Keep eval/prices; rewrite gates | 2–3 |
| Minimal-core research floor | 4 |
| Expanding F1–F3 + majority + official + F3 gate | 3 |
| Parallel R1 OLS + R2 mpairs | 1, 6, 7 |
| Lead-only promote; freeze teamName during research | 6–8 |
| First grids as specified | 6–7 |
| Protocol / agent docs | 5 |
| Default-off invariance / no mpairs dilution | 4 tests |
| Optional ensemble if both strong | Deferred (call out in Task 8 park) — YAGNI unless both promote |

No TBD placeholders. `simulate_range` / `mark_robust` / fold field names are consistent across tasks.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-13-walkforward-protocol.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks  
**2. Inline Execution** — execute tasks in this session with executing-plans checkpoints  

Which approach?
