"""Numerical check of the two discretization laws we are contributing, against the
exact tie-alphabet coordinates from gop_bias_theory (Gaussian source).

D-law   :  D0 - D(Delta) = c/(m-2)! + O(c^2)
H-law   :  (H(Delta)-H0) logM = C(m,2) [ c log(1/c) + c(1 - log m) ] + O(c^2)
           (leading term alone is C(m,2) c log(1/c); the second term is what we add,
            and it is what explains the slow approach to the leading coefficient)
"""
import numpy as np
from math import factorial, log, comb
import gop_bias_theory as T

def exact_coords(Delta, m):
    q = T.q_levels(Delta)
    P = T.make_P(q)
    s2, D, tot, H = T.s2_and_D(m, P)
    c = float((q ** 2).sum())
    return c, D, H

for m in (3, 4, 5):
    M = T.FUBINI[m]
    H0 = log(factorial(m)) / log(M)
    D0 = 1 / factorial(m) - 1 / M
    C2 = comb(m, 2)
    print(f"\n=== m={m}  (M={M}, H0={H0:.5f}, D0={D0:.6f}, C(m,2)={C2}, 1/(m-2)!={1/factorial(m-2):.4f}) ===")
    print(f"{'Delta':>7} {'c':>10} | {'D0-D':>11} {'c/(m-2)!':>11} {'ratio':>7} | "
          f"{'H-H0 (x logM)':>13} {'lead only':>11} {'lead+next':>11}")
    for Delta in (0.08, 0.04, 0.02, 0.01):
        c, D, H = exact_coords(Delta, m)
        dD = D0 - D
        pred_D = c / factorial(m - 2)
        lhsH = (H - H0) * log(M)
        lead = C2 * c * log(1 / c)
        leadnext = C2 * (c * log(1 / c) + c * (1 - log(m)))
        print(f"{Delta:>7} {c:>10.6f} | {dD:>11.3e} {pred_D:>11.3e} {dD/pred_D:>7.4f} | "
              f"{lhsH:>13.3e} {lead:>11.3e} {leadnext:>11.3e}")
    print("   (ratio -> 1 confirms the D-law; 'lead+next' should track 'H-H0 (x logM)'"
          " far better than 'lead only')")
