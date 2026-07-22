#!/usr/bin/env python
"""Walk-forward validation to test robustness across different time windows."""

import numpy as np
import pandas as pd
from teamName import getMyPosition as getPosition

nInst = 0
nt = 0

pricesFile = "./prices.txt"
scoreDefaultParam = 1.0
defaultCommRate = 0.0001
inst0CommRate = 0.00002
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000

def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    nt, nInst = df.shape
    return (df.values).T

def score(mu, sigma, param=scoreDefaultParam):
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac

def calcPL(prcHist, testStart, testEnd, window_name=""):
    """Calculate P&L for a specific test window."""
    commRate = np.full(nInst, defaultCommRate)
    commRate[0] = inst0CommRate
    dlrPosLimit = np.full(nInst, defaultDlrPosLimit)
    dlrPosLimit[0] = inst0DlrPosLimit
    
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    value = 0
    comm = 0
    
    todayPLL = []
    
    for t in range(testStart, testEnd + 1):
        prcHistSoFar = prcHist[:, :t]
        curPrices = prcHistSoFar[:, -1]

        if t < testEnd:
            newPosOrig = getPosition(prcHistSoFar)
            posLimits = (dlrPosLimit / curPrices).astype(int)
            newPos = np.clip(newPosOrig, -posLimits, posLimits).astype(int)
        else:
            newPos = np.array(curPos)

        deltaPos = newPos - curPos
        cash -= curPrices.dot(deltaPos) + comm
        dvolumes = curPrices * np.abs(deltaPos)
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume
        comm = np.sum(dvolumes * commRate)
            
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        value = cash + posValue

        if t > testStart:
            todayPLL.append(todayPL)
            
    pll = np.array(todayPLL)
    plmu, plstd = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if plstd > 0:
        annSharpe = np.sqrt(250) * plmu / plstd
    
    score_val = score(plmu, plstd, scoreDefaultParam)
    
    print(f"\n{window_name}")
    print(f"  Days {testStart}-{testEnd} ({testEnd - testStart} test days)")
    print(f"  Mean PL: ${plmu:.1f} | Sharpe: {annSharpe:.2f} | Score: ${score_val:.2f}")
    print(f"  Total Volume: ${totDVolume:,.0f}")
    
    return plmu, plstd, annSharpe, score_val

prcAll = loadPrices(pricesFile)
print(f"Loaded {nInst} instruments for {nt} days\n")
print("=" * 70)
print("WALK-FORWARD VALIDATION (Robustness Test)")
print("=" * 70)

# Current setup (what you optimized for)
calcPL(prcAll, 501, 750, "CURRENT (Optimized for)")

# Earlier windows (unseen data simulation)
calcPL(prcAll, 251, 500, "WINDOW 1 (More realistic)")
calcPL(prcAll, 201, 450, "WINDOW 2")
calcPL(prcAll, 101, 350, "WINDOW 3 (Oldest)")

print("\n" + "=" * 70)
print("If scores are similar across windows → algorithm is ROBUST")
print("If current score >> other windows → algorithm is OVERFITTED")
print("=" * 70)
