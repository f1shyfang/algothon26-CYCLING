import numpy as np
import time

# Create dummy data resembling our setup
# 51 instruments, 60 days lookback
nins = 51
lb = 40
np.random.seed(42)
rets = np.random.randn(nins - 1, lb)

VOLATILITY_FLOOR = 1e-12
mpairs_min_corr = 0.1 # lower to get candidates

# Method 1: Original loop-based corrcoef
t0 = time.time()
candidates_1 = []
n = rets.shape[0]
for i in range(n):
    for j in range(i + 1, n):
        a, b = rets[i], rets[j]
        if a.std() < VOLATILITY_FLOOR or b.std() < VOLATILITY_FLOOR:
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        if abs(corr) >= mpairs_min_corr:
            candidates_1.append((abs(corr), i + 1, j + 1))
candidates_1.sort(reverse=True)
t1 = time.time()
time_1 = t1 - t0

# Method 2: Vectorized corrcoef
t0 = time.time()
stds = np.std(rets, axis=1)
valid = stds >= VOLATILITY_FLOOR
corr_matrix = np.corrcoef(rets)
candidates_2 = []
for i in range(n):
    if not valid[i]:
        continue
    for j in range(i + 1, n):
        if not valid[j]:
            continue
        corr = corr_matrix[i, j]
        if np.isnan(corr):
            continue
        if abs(corr) >= mpairs_min_corr:
            candidates_2.append((abs(corr), i + 1, j + 1))
candidates_2.sort(reverse=True)
t2 = time.time()
time_2 = t2 - t0

print(f"Method 1 (original): {time_1:.4f} seconds, found {len(candidates_1)} candidates")
print(f"Method 2 (vectorized): {time_2:.4f} seconds, found {len(candidates_2)} candidates")
# Compare candidates up to float precision (float conversion can vary slightly)
equal = True
if len(candidates_1) != len(candidates_2):
    equal = False
else:
    for c1, c2 in zip(candidates_1, candidates_2):
        if abs(c1[0] - c2[0]) > 1e-12 or c1[1] != c2[1] or c1[2] != c2[2]:
            equal = False
            break
print(f"Equal results? {equal}")
print(f"Speedup: {time_1 / time_2:.1f}x")
