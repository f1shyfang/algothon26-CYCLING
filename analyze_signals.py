#!/usr/bin/env python
"""Analyze which signals are profitable across different windows."""

import numpy as np
import pandas as pd
from teamName import getMyPosition as getPosition

nInst = 0
nt = 0
pricesFile = "./prices.txt"
defaultCommRate = 0.0001
inst0CommRate = 0.00002
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000

def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    nt, nInst = df.shape
    return (df.values).T

def analyze_profitability(prcHist, testStart, testEnd, window_name=""):
    """Analyze daily P&L by instrument to see which are profitable."""
    commRate = np.full(nInst, defaultCommRate)
    commRate[0] = inst0CommRate
    dlrPosLimit = np.full(nInst, defaultDlrPosLimit)
    dlrPosLimit[0] = inst0DlrPosLimit
    
    curPos = np.zeros(nInst)
    daily_pnls = {i: [] for i in range(nInst)}
    positions = {i: [] for i in range(nInst)}
    
    print(f"\n{window_name} ({testStart}-{testEnd})")
    print("-" * 80)
    
    for t in range(testStart, testEnd + 1):
        prcHistSoFar = prcHist[:, :t]
        curPrices = prcHistSoFar[:, -1]

        if t < testEnd:
            newPosOrig = getPosition(prcHistSoFar)
            posLimits = (dlrPosLimit / curPrices).astype(int)
            newPos = np.clip(newPosOrig, -posLimits, posLimits).astype(int)
        else:
            newPos = np.array(curPos)

        # Mark-to-market P&L for each position
        if t > testStart:
            pnl_per_inst = curPos * (prcHistSoFar[:, -1] - prcHistSoFar[:, -2])
            for i in range(nInst):
                daily_pnls[i].append(pnl_per_inst[i])
                positions[i].append(curPos[i])

        curPos = np.array(newPos)
    
    # Aggregate stats per instrument
    inst_stats = []
    for i in range(nInst):
        if len(daily_pnls[i]) == 0:
            continue
        pnls = np.array(daily_pnls[i])
        pos = np.array(positions[i])
        
        total_pnl = np.sum(pnls)
        avg_pos = np.mean(np.abs(pos))
        times_traded = np.sum(pos[:-1] != pos[1:])  # position changes
        
        if total_pnl != 0:  # Only show profitable/loss-making instruments
            inst_stats.append({
                'instrument': i,
                'total_pnl': total_pnl,
                'avg_position': avg_pos,
                'times_traded': times_traded,
                'avg_daily_pnl': np.mean(pnls)
            })
    
    # Sort by profitability
    inst_stats.sort(key=lambda x: x['total_pnl'], reverse=True)
    
    print(f"\nTOP 10 PROFITABLE INSTRUMENTS:")
    print(f"{'Inst':<6} {'Total P&L':<12} {'Avg Pos':<12} {'Trades':<8} {'Avg Daily':<12}")
    
    top_pnl = 0
    for stat in inst_stats[:10]:
        print(f"{stat['instrument']:<6} ${stat['total_pnl']:<11.0f} {stat['avg_position']:<11.1f} {stat['times_traded']:<7} ${stat['avg_daily_pnl']:<11.2f}")
        top_pnl += stat['total_pnl']
    
    print(f"\nTOP 10 LOSERS:")
    print(f"{'Inst':<6} {'Total P&L':<12} {'Avg Pos':<12} {'Trades':<8} {'Avg Daily':<12}")
    
    for stat in reversed(inst_stats[-10:]):
        print(f"{stat['instrument']:<6} ${stat['total_pnl']:<11.0f} {stat['avg_position']:<11.1f} {stat['times_traded']:<7} ${stat['avg_daily_pnl']:<11.2f}")
    
    total_pnl = sum(s['total_pnl'] for s in inst_stats)
    avg_daily = total_pnl / (testEnd - testStart)
    print(f"\nTOTAL P&L: ${total_pnl:.0f} | AVG DAILY: ${avg_daily:.1f}")
    
    return {
        'window': window_name,
        'total_pnl': total_pnl,
        'profitable_instrs': [s['instrument'] for s in inst_stats[:10]],
        'losing_instrs': [s['instrument'] for s in inst_stats[-10:]]
    }

prcAll = loadPrices(pricesFile)
print(f"Loaded {nInst} instruments for {nt} days")
print("=" * 80)
print("SIGNAL PROFITABILITY ANALYSIS")
print("=" * 80)

result_current = analyze_profitability(prcAll, 501, 750, "CURRENT (501-750)")
result_window1 = analyze_profitability(prcAll, 251, 500, "WINDOW 1 (251-500)")

print("\n" + "=" * 80)
print("CONSISTENCY CHECK")
print("=" * 80)
current_profitable = set(result_current['profitable_instrs'])
window1_profitable = set(result_window1['profitable_instrs'])
overlap = current_profitable & window1_profitable

print(f"\nCurrent top 10 profitable: {sorted(result_current['profitable_instrs'])}")
print(f"Window 1 top 10 profitable: {sorted(result_window1['profitable_instrs'])}")
print(f"Overlap: {sorted(overlap)}")
print(f"Overlap ratio: {len(overlap)}/10 = {len(overlap)*10}%")

if len(overlap) < 3:
    print("\n⚠️  POOR OVERLAP - Algorithm is not generalizing!")
    print("   Different instruments are profitable in different periods.")
    print("   Strategy needs to be more selective or dynamic.")
elif len(overlap) >= 7:
    print("\n✓ GOOD OVERLAP - Algorithm has consistent profitable instruments")
    print("  Can focus optimization on these stable edges.")
