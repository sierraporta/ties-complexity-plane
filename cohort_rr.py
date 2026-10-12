"""Cohort RR analysis for the tie-aware ordinal plane application.

Two PhysioNet RR sources at different native lattices:
  - nsr2db   (Normal Sinus Rhythm RR Interval DB), ECG at 128 Hz -> 7.8125 ms lattice
  - fantasia (healthy young/elderly),              ECG at 250 Hz -> 4.0000 ms lattice

RR series are built from beat annotations, keeping normal-to-normal (NN) intervals only
(ectopic A/V and noise ~ beats dropped), with a physiological range filter [300,2000] ms and
a 20% relative-jump filter. We DROP offending intervals rather than impute them, because
median-imputation would manufacture ties and bias the very quantity under study.

Per record we report the tie fraction, the tie-aware GOP coordinates (m=4, M=75), and the
tie-broken index-vs-jitter entropy gap (M=24). The cohort shows that (i) the gap grows with the
tie fraction across records, and (ii) the two databases sit in different tie bands because their
lattices differ, i.e. the quantization step sets the tie fraction for a structural reason. The
tie-matched permutation null, read two-sided, distinguishes the observed series from its
surrogate in essentially every record.
"""
import numpy as np, wfdb, csv
from pathlib import Path
import apply_diagnostics as A
import gop_rr as R6

HERE = Path(__file__).resolve().parent
# Locate beat-annotation files by database key anywhere under the project (works with the
# wget layout under RealData/<...>/nsr2db/... and with a flat annot/<db>/ directory alike).
DATA = HERE / "RealData" if (HERE / "RealData").exists() else HERE

def ecg_files(dbkey):
    return sorted(p for p in DATA.rglob("*.ecg") if f"/{dbkey}/" in p.as_posix())

def record_path(dbkey, rec):
    for p in ecg_files(dbkey):
        if p.stem == rec:
            return p.with_suffix("")
    raise FileNotFoundError(f"{dbkey}/{rec}.ecg not found under {DATA}")

def rr_samples(path, fs):
    """NN RR intervals in integer samples from a WFDB beat-annotation file."""
    ann = wfdb.rdann(str(path), "ecg")
    sym = np.asarray(ann.symbol); samp = np.asarray(ann.sample)
    isN = sym == "N"
    good = isN[:-1] & isN[1:]            # both endpoints normal
    return np.diff(samp)[good].astype(np.int64)

def clean(rr, fs):
    """Physiological range [300,2000] ms + 20% relative-jump filter. Drops, never imputes."""
    lo, hi = 0.300 * fs, 2.000 * fs
    rr = rr[(rr >= lo) & (rr <= hi)]
    if len(rr) < 3:
        return rr
    rel = np.abs(np.diff(rr)) / rr[:-1]
    return rr[np.concatenate([[True], rel <= 0.20])]

def load_rr(path, fs):
    return clean(rr_samples(path, fs), fs)

def metrics(rr):
    m = 4
    g = R6.gop_coords(rr)                                   # tie-aware (M=75)
    Hi = A.coordinates(A.symbols(rr, m, 1, policy="index"), 24)["H"]
    Hj = A.coordinates(A.symbols(rr, m, 1, policy="jitter",
                                 rng=np.random.default_rng(0)), 24)["H"]
    return g, Hi, Hj

def pval_two(rr, cap=4000, B=299, seed=1):
    """Two-sided tie-matched permutation p on the score T."""
    x = rr[:cap] if len(rr) > cap else rr
    T0 = R6.gop_T(x); rng = np.random.default_rng(seed)
    Ts = np.array([R6.gop_T(rng.permutation(x)) for _ in range(B)])
    p_hi = (1 + np.sum(Ts >= T0)) / (B + 1)
    p_lo = (1 + np.sum(Ts <= T0)) / (B + 1)
    return min(1.0, 2 * min(p_hi, p_lo)), float(T0), float(Ts.mean())

DBS = [("nsr2db", 128, ecg_files("nsr2db")),
       ("fantasia", 250, ecg_files("fantasia"))]

def main():
    rows = []
    for db, fs, files in DBS:
        for f in files:
            rec = f.stem
            rr = load_rr(f.with_suffix(""), fs)
            if len(rr) < 500:
                print(f"skip {db}/{rec}: only {len(rr)} NN beats"); continue
            g, Hi, Hj = metrics(rr)
            p2, T0, Tn = pval_two(rr)
            rows.append(dict(db=db, rec=rec, fs=fs, n=len(rr), tie=g["tie"],
                             H=g["H"], D=g["D"], C=g["C_JS"], Dgap=g["D_gap"],
                             Hi=Hi, Hj=Hj, gap=Hi - Hj, p2=p2, T0=T0, Tnull=Tn))
            print(f"{db:9s} {rec:7s} n={len(rr):6d} tie={100*g['tie']:5.1f}%  "
                  f"H={g['H']:.3f} D={g['D']:.4f} C={g['C_JS']:.3f}  "
                  f"gap={Hi-Hj:+.4f}  p2={p2:.3f}  T0{'<' if T0<Tn else '>'}null")
    with open(HERE / "cohort_rr.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    tie = np.array([r["tie"] for r in rows]) * 100
    gap = np.array([r["gap"] for r in rows])
    isF = np.array([r["db"] == "fantasia" for r in rows])
    p2 = np.array([r["p2"] for r in rows])
    below = np.array([r["T0"] < r["Tnull"] for r in rows])
    print("\n==== COHORT SUMMARY ====")
    print(f"records: {len(rows)} (nsr2db {np.sum(~isF)}, fantasia {np.sum(isF)})")
    print(f"tie: nsr2db {tie[~isF].min():.0f}-{tie[~isF].max():.0f}% (med {np.median(tie[~isF]):.0f}), "
          f"fantasia {tie[isF].min():.0f}-{tie[isF].max():.0f}% (med {np.median(tie[isF]):.0f})")
    print(f"corr(tie, index-jitter gap) = {np.corrcoef(tie, gap)[0,1]:+.3f}")
    print(f"two-sided null rejects (p<0.05) in {np.sum(p2<0.05)}/{len(rows)} records; "
          f"observed flatter than surrogate in {np.sum(below)}/{len(rows)}")
    print("saved cohort_rr.csv")

if __name__ == "__main__":
    main()
