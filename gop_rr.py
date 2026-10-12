"""Generalized-ordinal-pattern (tie-aware) plane coordinates for a 1-D series, m=4.

The alphabet is the a(4)=75 weak orders; a window is coded by the sign pattern of its
pairwise differences (sign in {-1,0,+1}), so equal values are kept as ties. Returns the
normalized entropy H (base log 75), the disequilibrium D (plug-in, distinct-pairs U, and
gap with w=m-1), and the Jensen-Shannon complexity C_JS with the Q0 normalization. The
score T=(1-H)^2+C_JS^2 and a permutation (tie-matched) null are provided for inference.

Shared by the RR cohort scripts; contains no record-specific code.
"""
import numpy as np
from math import log
from itertools import combinations
from scipy.special import xlogy
import apply_diagnostics as A

m, Mgop = 4, 75
LOGM = log(Mgop)

def gop_codes(x):
    e = np.lib.stride_tricks.sliding_window_view(np.asarray(x, float), m)
    code = np.zeros(e.shape[0], dtype=np.int64)
    for p, (i, j) in enumerate(combinations(range(m), 2)):
        code += (np.sign(e[:, i] - e[:, j]).astype(np.int64) + 1) * (3 ** p)
    return code

def _q0(M):
    q = np.full(M, 1 / (2 * M)); q[0] = (1 + 1 / M) / 2
    return 1.0 / (-xlogy(q, q).sum() - 0.5 * log(M))
Q0 = _q0(Mgop)

def gop_coords(x):
    code = gop_codes(x)
    counts = np.unique(code, return_counts=True)[1].astype(float)
    n = counts.sum(); p = counts / n; K = len(counts)
    S = -xlogy(p, p).sum(); H = S / LOGM
    D = (p ** 2).sum() - 1 / Mgop
    Du = (counts * (counts - 1)).sum() / (n * (n - 1)) - 1 / Mgop
    qs = (p + 1 / Mgop) / 2
    J = (-xlogy(qs, qs).sum() + (Mgop - K) * (-(1 / (2 * Mgop)) * log(1 / (2 * Mgop)))
         - 0.5 * S - 0.5 * LOGM)
    C = Q0 * H * J
    c = np.bincount(code).astype(float); col = (c * (c - 1)).sum() / 2   # gap estimator w=m-1
    for h in range(1, m):
        col -= np.sum(code[:-h] == code[h:])
    N = code.shape[0]; Dg = col / ((N - m) * (N - m + 1) / 2) - 1 / Mgop
    return dict(H=H, D=D, D_U=Du, D_gap=Dg, C_JS=C, tie=A.tie_fraction(x, m, 1))

def gop_T(x):
    co = gop_coords(x); return (1 - co["H"]) ** 2 + co["C_JS"] ** 2

def perm_null_p(x, B, rng):
    """One-sided tie-matched permutation p: fraction of surrogates with score >= observed."""
    T0 = gop_T(x); cnt = 1
    for _ in range(B):
        cnt += gop_T(rng.permutation(x)) >= T0
    return cnt / (B + 1)
