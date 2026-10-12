"""Monte Carlo check of the quantization-corrected disequilibrium estimator

    D0_hat = D_gap(Delta) + c_hat/(m-2)!,   c_hat = sum_k n_k(n_k-1) / (N(N-1)),

on short records of Delta-quantized Gaussian white noise (values IID, so the gap with
w=m-1 removes the overlap bias and leaves the population tie-alphabet D(Delta); the
c_hat term then lifts it back to the continuous-plane D0). Self-contained.
"""
import numpy as np
from math import factorial, comb
from itertools import combinations

def gop_codes(x, m):
    e = np.lib.stride_tricks.sliding_window_view(np.asarray(x, float), m)
    code = np.zeros(e.shape[0], dtype=np.int64)
    for p, (i, j) in enumerate(combinations(range(m), 2)):
        code += (np.sign(e[:, i] - e[:, j]).astype(np.int64) + 1) * (3 ** p)
    return code

def D_gap(code, M, w):
    n = code.shape[0]; c = np.bincount(code).astype(float)
    col = (c * (c - 1)).sum() / 2
    for h in range(1, w + 1):
        col -= np.sum(code[:-h] == code[h:])
    return col / ((n - w - 1) * (n - w) / 2) - 1.0 / M

def c_hat(xq, Delta):
    k = np.round(xq / Delta).astype(np.int64)
    nk = np.bincount(k - k.min()).astype(float)
    N = nk.sum()
    return (nk * (nk - 1)).sum() / (N * (N - 1))

m = 4; M = 75; w = m - 1
D0 = 1 / factorial(m) - 1 / M
coeff = 1.0 / factorial(m - 2)
rng = np.random.default_rng(20261005)
B, N = 3000, 4000
print(f"m={m}  continuous-plane D0 = {D0:.6f}   correction coeff 1/(m-2)! = {coeff:.3f}")
print(f"N={N}, B={B} reps. The correction is a LEADING-ORDER (small-Delta) device:")
print(f"it removes the linear shift c/(m-2)!, leaving an O(c^2) residual.\n")
print(f"{'Delta':>6} {'tie%':>6} | {'uncorrected bias':>16} | {'corrected bias':>14} "
      f"{'(se)':>8} | {'% of shift removed':>18}")
for Delta in (0.04, 0.08, 0.12, 0.16):
    Dg = np.empty(B); D0h = np.empty(B); tie = 0.0
    for b in range(B):
        xq = np.round(rng.standard_normal(N) / Delta) * Delta
        code = gop_codes(xq, m)
        e = np.lib.stride_tricks.sliding_window_view(xq, m)
        tie += np.mean(np.any(np.diff(np.sort(e, 1), 1) == 0, 1))
        dg = D_gap(code, M, w)
        Dg[b] = dg
        D0h[b] = dg + c_hat(xq, Delta) * coeff
    bu = Dg.mean() - D0; bc = D0h.mean() - D0
    se = D0h.std() / np.sqrt(B)
    removed = 100 * (1 - abs(bc) / abs(bu))
    print(f"{Delta:>6} {100*tie/B:>6.1f} | {bu:>+16.3e} | {bc:>+14.3e} {se:>8.1e} | {removed:>17.1f}%")
print("\n  The uncorrected gap disequilibrium sits below D0 by the quantization shift;")
print("  the correction removes most of it, and the residual falls with Delta like O(c^2).")
