#!/usr/bin/env python
"""Algothon 2026 evaluation script with walk-forward validation.

Shows both local performance AND estimated out-of-sample performance.
"""

import numpy as np
import pandas as pd
from teamName import getMyPosition as getPosition

nInst = 0
nt = 0

pricesFile = "./prices.txt"
numTestDays = 250

# parameter for scoring function
scoreDefaultParam = 1.0

# commission rates (0.0001 = 1bp)
# SPECIAL rate for instrument 0
defaultCommRate = 0.0001
inst0CommRate = 0.00002

# position limits (dollars)
# SPECIAL position limit for instrument 0
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000

def loadPrices(fn):
    """
    Load prices from csv file (one instrument per column) and transpose into one instrument per row
    """
    global nt, nInst
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    nt, nInst = df.shape
    return (df.values).T


def chargeFees(dvolumes, commRate):
    """
    Total commission for one day's trades.
    """
    return np.sum(dvolumes * commRate)


def score(mu, sigma, param=scoreDefaultParam):
    """
    Final score from the daily-PnL mean & std, plus a scoring parameter.
    """
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac

prcAll = loadPrices(pricesFile)
print(f"Loaded {nInst} instruments for {nt} days")

# initialise the per-instrument commissions and position limits
commRate = np.full(nInst, defaultCommRate)
commRate[0] = inst0CommRate
dlrPosLimit = np.full(nInst, defaultDlrPosLimit)
dlrPosLimit[0] = inst0DlrPosLimit

def calcPL(prcHist, testStart, testEnd, verbose=False):
    """
    Calculate P&L for a specific test window.
    """
    
    # initial values
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    value = 0
    comm = 0
    
    todayPLL = []
    _, nt = prcHist.shape
    
    for t in range(testStart, testEnd + 1):
        # price history up to and including t
        prcHistSoFar = prcHist[:, :t]
        curPrices = prcHistSoFar[:, -1]

        # trading loop, do not do it on the very last day of the test
        if t < testEnd:
            # get new positions
            newPosOrig = getPosition(prcHistSoFar)

            # clip to position limits, and enforce integer shares
            posLimits = (dlrPosLimit / curPrices).astype(int)
            newPos = np.clip(newPosOrig, -posLimits, posLimits).astype(int)
        else:
            # the final day is only used as 'mark' of final PnL
            newPos = np.array(curPos)

        # change in positions
        deltaPos = newPos - curPos
        
        cash -= curPrices.dot(deltaPos) + comm

        # calculate commissions
        dvolumes = curPrices * np.abs(deltaPos)
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume
        comm = chargeFees(dvolumes, commRate)
            
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        # PnL is the daily change in portfolio value (cash plus positions)
        todayPL = cash + posValue - value
        
        value = cash + posValue

        # only score for test days
        if t > testStart:
            if verbose:
                ret = 0.0
                if totDVolume > 0:
                    ret = value / totDVolume
                print(
                    f"Day {t} value: {value:.2f} todayPL: ${todayPL:.2f} $-traded: {totDVolume:.0f} return: {ret:.5f}"
                )
            todayPLL.append(todayPL)
            
    pll = np.array(todayPLL)
    plmu, plstd = (np.mean(pll), np.std(pll))

    # calculate annualised Sharpe
    annSharpe = 0.0
    if plstd > 0:
        annSharpe = np.sqrt(250) * plmu / plstd
    
    score_val = score(plmu, plstd, scoreDefaultParam)
    ret = 0.0
    if totDVolume > 0:
        ret = value / totDVolume
        
    return {
        'mean_pl': plmu,
        'return': ret,
        'std_pl': plstd,
        'sharpe': annSharpe,
        'dvol': totDVolume,
        'score': score_val
    }

print("=" * 80)
print("WALK-FORWARD VALIDATION (Out-of-Sample Robustness Check)")
print("=" * 80)

# Test on current window (what you see locally)
result_current = calcPL(prcAll, 501, 750, verbose=True)
print("\n" + "=" * 80)
print("LOCAL WINDOW (501-750) - What you see locally:")
print("=" * 80)
print(f"mean(PL): ${result_current['mean_pl']:.1f}")
print(f"return: {result_current['return']:.5f}")
print(f"StdDev(PL): ${result_current['std_pl']:.2f}")
print(f"annSharpe(PL): {result_current['sharpe']:.2f}")
print(f"totDvolume: ${result_current['dvol']:.0f}")
print(f"Score: ${result_current['score']:.2f}")

# Test on earlier windows (simulate generalization)
print("\n" + "=" * 80)
print("EARLIER WINDOWS (simulate unseen data like website would see):")
print("=" * 80)

results = {'current': result_current}

for name, (start, end) in [
    ("Window 1 (251-500)", (251, 500)),
    ("Window 2 (201-450)", (201, 450)),
    ("Window 3 (101-350)", (101, 350)),
]:
    result = calcPL(prcAll, start, end, verbose=False)
    results[name] = result
    print(f"\n{name}:")
    print(f"  Mean PL: ${result['mean_pl']:.1f} | Sharpe: {result['sharpe']:.2f} | Score: ${result['score']:.2f}")

# Estimate out-of-sample score
scores_earlier = [results[k]['score'] for k in results if k != 'current']
avg_earlier = np.mean(scores_earlier)
median_earlier = np.median(scores_earlier)

print("\n" + "=" * 80)
print("GENERALIZATION ASSESSMENT:")
print("=" * 80)
print(f"Local window score:      ${result_current['score']:.2f}")
print(f"Earlier windows (avg):   ${avg_earlier:.2f}")
print(f"Earlier windows (median):${median_earlier:.2f}")
print(f"\nEstimated website score: ~${median_earlier:.0f} (based on historical patterns)")

if result_current['score'] > avg_earlier * 3:
    print("\n⚠️  SEVERE OVERFITTING DETECTED")
    print("   Your local score is 3x+ higher than earlier windows.")
    print("   Website will likely score significantly lower.")
    print(f"   Expect ~${median_earlier:.0f} on unseen data, not ${result_current['score']:.0f}")
elif result_current['score'] > avg_earlier * 1.5:
    print("\n⚠️  MODERATE OVERFITTING")
    print("   Your local window may be easier than average.")
    print(f"   Expect ~${median_earlier:.0f} on unseen data.")
else:
    print("\n✓ GOOD GENERALIZATION")
    print("   Algorithm is consistent across windows.")
