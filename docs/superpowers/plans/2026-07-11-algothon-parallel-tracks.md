# Algothon Parallel Tracks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a 4-day parallel-agent search across cross-sectional, pairs, ALGO-hedge, and ensemble strategy families, promoting only robust score gains into `teamName.py` for leaderboard submission.

**Architecture:** Keep production `teamName.py` as the floor (current official score **207.37**: 5d/20d reversal, band=0.195, regime 10/60@1.15×0.32). Extend `loop.py` Params with default-off knobs per track so agents A/B/C research in parallel without colliding. Lead merges at most one promote per checkpoint via `python loop.py --json`. Track D ensembles only survivors with positive standalone edge and low PnL correlation.

**Tech Stack:** Python 3, numpy, pandas, unittest, existing `eval.py` / `loop.py` harness, Cursor parallel agents.

**Spec:** `docs/superpowers/specs/2026-07-11-algothon-parallel-tracks-design.md`

---

## File map

| File | Responsibility |
|:---|:---|
| `teamName.py` | Production submission logic only (lead edits on promote) |
| `loop.py` | Parametrised research harness, grids, promote verdict |
| `eval.py` | Official scorer — **never edit** |
| `prices.txt` | Price data — **never edit** |
| `test_teamName.py` | Contract tests for production |
| `STRATEGY_LOOP.md` | Shared iteration log + backtest table |
| `AGENT_LOOP.md` | Per-tick agent protocol |
| `docs/tracks/A.md` … `D.md` | Per-track hypothesis banks + reject memory |
| `docs/superpowers/plans/2026-07-11-algothon-parallel-tracks.md` | This plan |

---

### Task 1: Sync harness baseline to production

**Files:**
- Modify: `loop.py` (`Params` defaults + `build_grid` comment)
- Test: run `python loop.py` and `python eval.py`

- [ ] **Step 1: Confirm production constants**

Open `teamName.py` and verify these values (as of plan write):

```python
LOOKBACKS = (5, 20)
REBALANCE_BAND = 0.195
REGIME_VOL_SHORT = 10
REGIME_VOL_LONG = 60
REGIME_THRESHOLD = 1.15
REGIME_SCALE = 0.32
```

- [ ] **Step 2: Align `Params` defaults in `loop.py`**

Ensure `Params` matches production exactly:

```python
@dataclass(frozen=True)
class Params:
    lookbacks: tuple[int, ...] = (5, 20)
    weights: tuple[float, ...] | None = None
    rebalance_band: float = 0.195
    algo_dollar_limit: float = 100_000.0
    default_dollar_limit: float = 10_000.0
    signal_clip: float = 1.0
    momentum_lookback: int = 10
    momentum_weight: float = 0.0
    regime_vol_short: int = 10
    regime_vol_long: int = 60
    regime_threshold: float = 1.15
    regime_scale: float = 0.32
```

- [ ] **Step 3: Verify harness matches official score**

Run:

```bash
python eval.py 2>&1 | tail -5
python loop.py 2>&1 | tail -20
```

Expected: both report **Score: 207.37** (or identical to within 0.01). If they disagree, fix `loop.py` `strategy_positions` until it matches — do not change `eval.py`.

- [ ] **Step 4: Commit sync**

```bash
git add loop.py teamName.py
git commit -m "chore: sync loop Params defaults to production baseline"
```

---

### Task 2: Create per-track docs and agent prompts

**Files:**
- Create: `docs/tracks/A.md`, `docs/tracks/B.md`, `docs/tracks/C.md`, `docs/tracks/D.md`

- [ ] **Step 1: Write Track A doc**

Create `docs/tracks/A.md`:

```markdown
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
```

- [ ] **Step 2: Write Track B doc**

Create `docs/tracks/B.md`:

```markdown
# Track B — Pairs / cointegration

**Status:** active  
**Owner:** Agent B  
**Params prefix:** `pairs_*` (default off)  
**Kill rule:** 3 consecutive non-promotable ticks → freeze

## Hypothesis bank
1. Top correlated pairs (ρ>0.7 on train window) trade spread z-score.
2. ALGO vs equal-weight basket residual mean reversion.
3. Small number of pairs (≤5) to limit turnover.

## Reject memory
(none yet)

## Log
| Tick | Hypothesis | Best score | train | half2 | Verdict |
|:---|:---|---:|---:|---:|:---|
```

- [ ] **Step 3: Write Track C doc**

Create `docs/tracks/C.md`:

```markdown
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
```

- [ ] **Step 4: Write Track D doc**

Create `docs/tracks/D.md`:

```markdown
# Track D — Ensemble blender

**Status:** idle until Day 3  
**Owner:** Agent D / lead  
**Rule:** Only blend tracks with standalone score > 0, robust=true, and PnL corr < 0.5 vs production.

## Candidates
(empty until A/B/C produce survivors)

## Log
| Tick | Blend | Best score | train | half2 | Verdict |
|:---|:---|---:|---:|---:|:---|
```

- [ ] **Step 5: Commit track docs**

```bash
git add docs/tracks/
git commit -m "docs: add parallel track hypothesis banks A–D"
```

---

### Task 3: Add Track A Params (cross-sectional, default off)

**Files:**
- Modify: `loop.py` (`Params`, `strategy_positions`, `build_grid`, `label`)
- Test: `python -c` smoke + `python loop.py --json` with empty xs weight

- [ ] **Step 1: Extend `Params` with xs fields**

Add to `Params` (defaults keep baseline identical):

```python
    # Track A: cross-sectional overlay (default off)
    xs_lookback: int = 5
    xs_weight: float = 0.0  # 0 => disabled
    xs_include_algo: bool = False
```

Update `label()` to append `xs=...` when `xs_weight > 0`.

- [ ] **Step 2: Implement xs signal in `strategy_positions`**

After the time-series (+ optional momentum) signal is built, before dollar sizing:

```python
    if params.xs_weight > 0:
        lb = params.xs_lookback
        if nt > lb:
            xs_ret = log_prices[:, -1] - log_prices[:, -(lb + 1)]
            if not params.xs_include_algo:
                xs_ret = xs_ret.copy()
                xs_ret[0] = np.nan
            # Cross-sectional demean (nan-safe)
            mu = np.nanmean(xs_ret)
            xs_signal = -(xs_ret - mu)  # reverse relative winners
            xs_signal = np.nan_to_num(xs_signal, nan=0.0)
            # Standardise cross-sectionally
            sd = np.std(xs_signal) + VOLATILITY_FLOOR
            xs_signal = np.clip(xs_signal / sd, -params.signal_clip, params.signal_clip)
            signal = (1.0 - params.xs_weight) * signal + params.xs_weight * xs_signal
```

- [ ] **Step 3: Smoke-test default-off invariance**

```bash
python - <<'PY'
from loop import Params, load_prices, evaluate
prc = load_prices()
base = evaluate(prc, Params())
off = evaluate(prc, Params(xs_weight=0.0, xs_lookback=10))
assert abs(base.score - off.score) < 1e-9
print("ok", round(base.score, 2))
PY
```

Expected: `ok 207.37`

- [ ] **Step 4: Add a Track A research grid helper (do not promote yet)**

Replace or extend `build_grid()` so agents can call a focused grid. Minimal version for Day 1:

```python
def build_grid() -> list[Params]:
    grid: list[Params] = [Params()]
    # Track A screen (xs overlay)
    for lb, w in ((5, 0.10), (5, 0.20), (10, 0.10), (10, 0.20)):
        grid.append(Params(xs_lookback=lb, xs_weight=w))
        grid.append(Params(xs_lookback=lb, xs_weight=w, xs_include_algo=True))
    return grid
```

- [ ] **Step 5: Run sweep (research only)**

```bash
python loop.py --sweep --csv results.csv
python loop.py --json
```

Expected: baseline ~207.37; `promote` true only if a candidate clears guards. Record outcomes in `docs/tracks/A.md` and `STRATEGY_LOOP.md`.

- [ ] **Step 6: Commit Track A plumbing**

```bash
git add loop.py docs/tracks/A.md STRATEGY_LOOP.md results.csv
git commit -m "feat: add default-off cross-sectional overlay Params for track A"
```

---

### Task 4: Add Track B Params (pairs residual, default off)

**Files:**
- Modify: `loop.py`
- Update: `docs/tracks/B.md`

- [ ] **Step 1: Extend `Params`**

```python
    # Track B: ALGO-vs-basket residual (simple pairs proxy; default off)
    pairs_lookback: int = 20
    pairs_weight: float = 0.0  # 0 => disabled
    pairs_entry_z: float = 1.5
```

- [ ] **Step 2: Implement residual overlay**

Cheap, submission-safe pairs proxy (no statsmodels required in production later):

```python
    if params.pairs_weight > 0 and nt > params.pairs_lookback:
        lb = params.pairs_lookback
        # Equal-weight basket of instruments 1..n (ex-ALGO)
        basket = np.nanmean(log_prices[1:, :], axis=0)
        algo = log_prices[0, :]
        spread = algo - basket
        # Rolling z on the available window in `recent`
        mu_s = spread[-lb:].mean()
        sd_s = spread[-lb:].std() + VOLATILITY_FLOOR
        z = (spread[-1] - mu_s) / sd_s
        # Mean-revert ALGO vs basket: if ALGO rich (z>0), short ALGO / long basket
        if abs(z) >= params.pairs_entry_z:
            pair_sig = np.zeros(nins)
            pair_sig[0] = -np.clip(z, -params.signal_clip, params.signal_clip)
            # Dollar-neutral-ish: distribute opposite among others
            pair_sig[1:] = -pair_sig[0] / (nins - 1)
            signal = (1.0 - params.pairs_weight) * signal + params.pairs_weight * pair_sig
```

- [ ] **Step 3: Default-off invariance test**

```bash
python - <<'PY'
from loop import Params, load_prices, evaluate
prc = load_prices()
base = evaluate(prc, Params())
off = evaluate(prc, Params(pairs_weight=0.0))
assert abs(base.score - off.score) < 1e-9
print("ok", round(base.score, 2))
PY
```

- [ ] **Step 4: Track B grid + sweep**

Temporarily set `build_grid()` to:

```python
def build_grid() -> list[Params]:
    grid = [Params()]
    for w, z in ((0.10, 1.0), (0.10, 1.5), (0.20, 1.5), (0.20, 2.0)):
        grid.append(Params(pairs_weight=w, pairs_entry_z=z, pairs_lookback=20))
        grid.append(Params(pairs_weight=w, pairs_entry_z=z, pairs_lookback=40))
    return grid
```

Run `python loop.py --sweep` and `python loop.py --json`. Log in `docs/tracks/B.md`.

- [ ] **Step 5: Commit**

```bash
git add loop.py docs/tracks/B.md STRATEGY_LOOP.md
git commit -m "feat: add default-off ALGO-basket residual overlay for track B"
```

---

### Task 5: Add Track C research knobs (ALGO hedge / scale)

**Files:**
- Modify: `loop.py`
- Update: `docs/tracks/C.md`

- [ ] **Step 1: Extend `Params`**

```python
    # Track C: ALGO-specific exposure multiplier (1.0 => production)
    algo_signal_scale: float = 1.0
    # Optional: hedge basket beta with ALGO (0 => off)
    algo_hedge_weight: float = 0.0
```

- [ ] **Step 2: Apply after signal computed**

```python
    if params.algo_signal_scale != 1.0:
        signal = signal.copy()
        signal[0] *= params.algo_signal_scale

    if params.algo_hedge_weight > 0:
        # Hedge: set ALGO to offset average non-ALGO signal
        basket_sig = float(np.mean(signal[1:]))
        signal = signal.copy()
        signal[0] = signal[0] * (1.0 - params.algo_hedge_weight) - params.algo_hedge_weight * basket_sig
```

- [ ] **Step 3: Default-off invariance**

```bash
python - <<'PY'
from loop import Params, load_prices, evaluate
prc = load_prices()
base = evaluate(prc, Params())
off = evaluate(prc, Params(algo_signal_scale=1.0, algo_hedge_weight=0.0))
assert abs(base.score - off.score) < 1e-9
print("ok", round(base.score, 2))
PY
```

- [ ] **Step 4: Track C grid + sweep**

```python
def build_grid() -> list[Params]:
    grid = [Params()]
    for s in (0.5, 0.75, 1.25, 1.5, 2.0):
        grid.append(Params(algo_signal_scale=s))
    for h in (0.25, 0.50, 0.75, 1.0):
        grid.append(Params(algo_hedge_weight=h))
    return grid
```

Run sweep/json; log in `docs/tracks/C.md`. Do **not** retest `$50K` caps (already rejected).

- [ ] **Step 5: Commit**

```bash
git add loop.py docs/tracks/C.md STRATEGY_LOOP.md
git commit -m "feat: add ALGO scale/hedge Params for track C"
```

---

### Task 6: Day 1–2 parallel agent execution protocol

**Files:**
- Modify: `docs/tracks/*.md`, `STRATEGY_LOOP.md`, optionally `loop.py` `build_grid`
- Modify production **only** via Task 8 promote procedure

- [ ] **Step 1: Morning kickoff (lead)**

1. Confirm baseline still 207.37: `python eval.py 2>&1 | tail -3`
2. Assign one hypothesis from each track bank to agents A, B, C.
3. Each agent gets this prompt skeleton:

```text
You are Track {A|B|C}. Follow AGENT_LOOP.md for one iteration.
Read docs/tracks/{A|B|C}.md reject memory — do not repeat exact rejects.
Set build_grid() ONLY for your Params prefix; leave other track weights at 0.
Never edit teamName.py or eval.py.
Run python loop.py --sweep and python loop.py --json.
Append results to docs/tracks/{X}.md and STRATEGY_LOOP.md.
Stop after one iteration with: hypothesis, verdict, next hypothesis.
```

- [ ] **Step 2: Launch 3 agents in parallel**

Use Cursor Task/subagents (or three chat sessions). Each runs one tick, then stops.

- [ ] **Step 3: Midday merge review (lead)**

For each track:

```bash
python loop.py --json
```

- If all `promote: false` → no `teamName.py` edit.
- If one `promote: true` → proceed to Task 8 for that winner only.
- If two claim promote → keep higher official score that still passes `half2 >= 0.95 * baseline half2`; park the other in `docs/tracks/D.md` candidates.

- [ ] **Step 4: Apply kill criteria**

If a track has 3 consecutive rejects, set `Status: frozen` in its doc and reassign that agent to another bank idea or to idle for Day 3 ensemble.

- [ ] **Step 5: Evening log**

Append one summary line to `STRATEGY_LOOP.md` Iteration Log covering all three ticks. Optional: one leaderboard submission if the live stage accepts entries (≤1/day).

- [ ] **Step 6: Repeat for remaining Day 1–2 ticks**

Target ~6–8 ticks/track across Days 1–2. Keep `build_grid` single-track-focused per agent to avoid cross-contamination.

---

### Task 7: Track D ensemble (Day 3)

**Files:**
- Modify: `loop.py` (ensemble Params)
- Modify: `docs/tracks/D.md`
- Possibly promote via Task 8

- [ ] **Step 1: Collect survivors**

List every Track A/B/C candidate that had `robust: true` and score within 10% of baseline even if not promoted. Prefer PnL correlation check:

```bash
python - <<'PY'
import numpy as np
from loop import Params, load_prices, simulate, NUM_TEST_DAYS

prc = load_prices()
base = simulate(prc, Params(), NUM_TEST_DAYS).pll

# Fill in survivor Params from track logs:
survivors = {
    "xs": Params(xs_lookback=5, xs_weight=0.10),  # replace with real survivors
}
for name, p in survivors.items():
    pll = simulate(prc, p, NUM_TEST_DAYS).pll
    corr = float(np.corrcoef(base, pll)[0, 1])
    print(name, "corr", round(corr, 3), "score", round(simulate(prc, p, NUM_TEST_DAYS).score, 2))
PY
```

Keep only survivors with `corr < 0.5` (or the lowest available if all higher — then ensemble is likely not worth it; skip D).

- [ ] **Step 2: Add ensemble Params**

```python
    # Track D: blend production signal with enabled overlays already in Params
    # (xs_weight / pairs_weight / algo_hedge_weight). No new math required if
    # overlays compose in strategy_positions. Use build_grid to search weights.
```

If overlays already compose when multiple weights > 0, Day 3 grid is just joint weight search:

```python
def build_grid() -> list[Params]:
    grid = [Params()]
    # Example — replace with actual survivor knobs from logs
    for xs, pw in ((0.05, 0.05), (0.10, 0.05), (0.10, 0.10), (0.0, 0.10)):
        grid.append(Params(xs_weight=xs, xs_lookback=5, pairs_weight=pw, pairs_lookback=20))
    return grid
```

- [ ] **Step 3: Sweep + json verdict**

```bash
python loop.py --sweep --csv results.csv
python loop.py --json
```

Log in `docs/tracks/D.md`. Promote only via Task 8.

- [ ] **Step 4: Commit research state**

```bash
git add loop.py docs/tracks/ STRATEGY_LOOP.md results.csv
git commit -m "research: day3 ensemble grid from track survivors"
```

---

### Task 8: Promote procedure (lead only, any day)

**Files:**
- Modify: `teamName.py` (constants / logic mirroring winning `Params`)
- Verify: `eval.py`, `test_teamName.py`
- Update: `STRATEGY_LOOP.md`

- [ ] **Step 1: Confirm machine verdict**

```bash
python loop.py --json
```

Proceed only if `"promote": true`. Copy the winning params from the JSON `winner` object.

- [ ] **Step 2: Port winner into `teamName.py`**

Update constants and, if a new overlay won, port the corresponding signal block from `loop.py` `strategy_positions` into `getMyPosition`, keeping the file self-contained (prefer `import numpy` only unless wiki allows more).

Example for a pure constant promote (band/regime):

```python
REBALANCE_BAND = 0.200  # example — use actual winner
REGIME_SCALE = 0.30     # example — use actual winner
```

- [ ] **Step 3: Confirm official score + tests**

```bash
python eval.py 2>&1 | tail -8
python -m unittest test_teamName.py -v
```

Expected: `eval.py` score matches `loop.py` winner score; all tests PASS. If mismatch → **revert** `teamName.py` and log as harness drift / REJECT.

- [ ] **Step 4: Update baseline in `loop.py` `Params` defaults** to the new production values so future sweeps compare against the new floor.

- [ ] **Step 5: Log + commit**

Append Iteration entry to `STRATEGY_LOOP.md` with Decision `KEEP & HARDEN`.

```bash
git add teamName.py loop.py STRATEGY_LOOP.md test_teamName.py
git commit -m "feat: promote robust strategy params to production"
```

---

### Task 9: Day 4 freeze, stress, submit

**Files:**
- Create: submission copy `<YourTeamName>.py` (or wiki-required name)
- Read: https://wiki.algothon.au/submission/

- [ ] **Step 1: Freeze**

No new strategy families. Production = last promoted `teamName.py`.

- [ ] **Step 2: Stress checks (research only)**

Document in `STRATEGY_LOOP.md` (do not need code changes if already covered):

- Score at 2× commissions (manual note from prior iteration 9 pattern, or temporary simulate tweak in a scratch run — do not commit eval changes).
- Confirm positions finite/integral under ±5% synthetic gap on last day (quick script ok).

- [ ] **Step 3: Submission contract audit**

```bash
python -m unittest test_teamName.py -v
python eval.py 2>&1 | tail -8
rg -n "^(import|from) " teamName.py
```

Checklist:

- [ ] Score positive and matches last promote
- [ ] Imports allowed for grading
- [ ] No file I/O / network
- [ ] `getMyPosition(prcSoFar)` → shape `(51,)`, ints, dollar limits

- [ ] **Step 4: Package**

```bash
cp teamName.py YourTeamName.py   # use registered team name / wiki filename
# zip per Submission Guide — only algorithm file (+ requirements.txt if needed)
```

- [ ] **Step 5: Submit**

Submit via https://www.algothon.au/leaderboard (≤1/day). Record timestamp + score claim in `STRATEGY_LOOP.md`.

- [ ] **Step 6: 16 Jul restart playbook**

When General Round `prices.txt` drops:

1. Replace `prices.txt`.
2. `python eval.py` → new baseline number.
3. Sync `Params` defaults; reset track logs’ “next hypothesis” but keep reject memory.
4. Restart Tasks 6–8 on the new data.

```bash
git add YourTeamName.py STRATEGY_LOOP.md
git commit -m "chore: freeze submission copy and day4 checklist notes"
```

---

## Agent launch cheat-sheet (copy/paste)

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

## Self-review (plan vs spec)

| Spec requirement | Task |
|:---|:---|
| Parallel tracks A/B/C/D | Tasks 2–7 |
| Promote gate / lead-only merge | Tasks 6, 8 |
| Default-off research Params | Tasks 3–5 |
| 4-day schedule | Tasks 6, 7, 9 |
| Kill after 3 rejects | Task 6 Step 4 |
| Submission checklist | Task 9 |
| 16 Jul restart | Task 9 Step 6 |
| Never edit eval/prices | Stated in Tasks 1, 6, 8 |

No TBD placeholders. Params field names are consistent across tasks (`xs_*`, `pairs_*`, `algo_signal_scale`, `algo_hedge_weight`).
