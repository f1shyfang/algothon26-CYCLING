# Agent Loop — Autonomous Strategy Improvement

> A self-contained protocol an AI agent executes to iterate on `getMyPosition`
> and **only** ship changes that improve the official score without overfitting.
> Hand this whole file to an agent as the prompt. It runs **one iteration per
> invocation**, then stops. To run it on a schedule, see [Running on a loop](#running-on-a-loop).

---

## Objective

Maximise the Algothon score on the last 250 days of `prices.txt`:
`score = μ × SR² / (SR² + 1)`, where `SR = √250 · μ/σ` of daily PnL.
Research baseline (`Params()` minimal core in `loop.py`): **score = 211.49**
(5d/20d volatility-standardised reversal, 19.5% rebalance band, high-vol
regime cut 10d/60d @ 1.15 → 22% exposure; all overlays off). Live
`teamName.py` may still score **265.18** until the lead promotes — do not
treat `eval.py` as the research floor during walk-forward ticks.

The agent's job is to **raise the baseline** — or prove it can't be beaten this
round and say so. A non-improving iteration that is honestly rejected is a
success, not a failure.

---

## Hard guardrails (never violate)

1. **Never edit `eval.py`, `prices.txt`, or `loop.py`'s simulator/scoring.** Those define ground truth.
2. **`teamName.py` stays submission-safe:** only `import numpy`; no file I/O, no
   network, no packages outside `requirements-dev.txt`, signature exactly
   `getMyPosition(prcSoFar) -> np.ndarray` of shape `(51,)`, integer positions.
3. **No look-ahead:** signals may use only `prcSoFar` (history up to today).
4. **Promote only on the deterministic verdict** from `loop.py --json` (see
   step 4). Do not promote on official score alone — overfitting is the default
   failure.
5. **One change per iteration.** Isolate the variable so the log stays causal.
6. **Never fabricate results.** Every number in the log must come from a real run.

---

## The iteration (run these steps in order, once)

### 1 — Read state
- Read the **Iteration Log** in `STRATEGY_LOOP.md` (what's been tried + rejected).
  Do **not** re-propose a rejected idea unless you change what made it fail.
- Read `teamName.py` (current production logic) and `loop.py` (`Params`, `build_grid`).

### 2 — Form ONE hypothesis
- Pick the single most promising untried idea. State it in the template:
  > "**[instruments]** exhibit **[property]** over **[timeframe]**, exploitable by
  > **[action]**; expected to raise score because **[mechanism]**."
- Prefer ideas expressible as `Params` (lookbacks, weights, band, ALGO cap,
  signal clip). If it needs new logic, add a parameterised variant to
  `strategy_positions` in `loop.py` **behind a new `Params` field** (default
  off, so the baseline is unchanged), then extend `build_grid`.

### 3 — Evaluate (research only — do NOT touch `teamName.py` yet)
- Add the candidate(s) to `build_grid()` in `loop.py`.
- Run the sweep and inspect robustness:

```bash
python loop.py --sweep
```

- Every candidate is scored on four windows: **official** (last 250d) and three
  **walk-forward folds** (F1/F2/F3: three consecutive 100-day segments ending at
  the official window). A win on `score` with a collapsed F3 or a minority of
  folds beating baseline is overfitting — expect it and reject it.

### 4 — Decide (deterministic)
Get the machine verdict instead of eyeballing:

```bash
python loop.py --json
```

- `promote: false` → **REJECT.** Record why (which gate failed), go to step 6.
- `promote: true` → **PROMOTE** the `winner` params (lead only — see
  `docs/tracks/PROTOCOL.md`). It has already passed all walk-forward gates:
  **majority of F1–F3 beat baseline**, **official score ↑**, and **F3 ≥
  0.95× baseline F3**.

### 5 — Promote (only if verdict said so)
- Edit **`teamName.py`** to match the winning `Params` (change the constants:
  `LOOKBACKS`, `REBALANCE_BAND`, etc.). Keep it self-contained.
- **Confirm with the official evaluator and tests — both must pass:**

```bash
python eval.py          # score must match loop.py's winner score
python -m unittest test_teamName.py -v   # all tests green
```

- If `eval.py` disagrees with `loop.py` or any test fails, **revert
  `teamName.py`** and treat the iteration as REJECT (and file the discrepancy
  in the log — the harness may have drifted from ground truth).

### 6 — Record & stop
- Append a numbered entry to the **Iteration Log** in `STRATEGY_LOOP.md`:
  Date · Hypothesis · Strategy · Result (real numbers) · Decision
  (`KEEP & HARDEN` / `DISCARD` / `FINAL`) · Learnings.
- Add the run to the results table in the **Backtest** section.
- End the iteration with a 2–3 line summary: hypothesis, verdict, new baseline
  score (unchanged if rejected), and the single best next hypothesis to try.

---

## Stopping criteria

Declare the loop **DONE** (decision `FINAL`) when any holds:
- 3 consecutive iterations produce no promotable candidate, **and** a
  parameter-sensitivity sweep (±30% around production) shows a stable plateau; or
- The remaining hypotheses in `STRATEGY_LOOP.md`'s bank are exhausted or already
  rejected with sound reasoning; or
- The user says stop / the submission deadline is reached.

On DONE: confirm `python eval.py` + tests pass, and that `teamName.py` is the
best verified strategy.

---

## Running on a loop

Execute one iteration per invocation. To automate, use the `/loop` skill with
this file as the prompt and a dynamic (self-paced) cadence — each iteration is a
few seconds of compute but a meaningful unit of reasoning, so pace by
iterations, not wall-clock:

```
/loop iterate on the strategy using AGENT_LOOP.md, one iteration per tick
```

Between ticks the agent should pick a *different* hypothesis each time (step 2's
"don't repeat rejected ideas" rule prevents thrashing). Stop the loop when a
stopping criterion is met.

---

## Why this converges (not just churns)

- **Ground truth is frozen** (`eval.py`) and the harness mirrors its scoring
  (`loop.py` reproduces the research floor **211.49** on minimal-core `Params()`),
  so score deltas are trustworthy.
- **The promote gate requires walk-forward survival** (majority folds + official
  ↑ + F3 ≥ 0.95×base F3), so the loop can only ratchet the baseline *up* on
  genuinely robust edges.
- **The log is the memory**: every rejection encodes a constraint, so the search
  space shrinks each pass instead of revisiting dead ends.
