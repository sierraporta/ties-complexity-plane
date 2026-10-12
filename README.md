# Ordinal complexity–entropy plane with ties: a study of quantization in short series

Working repository for an ongoing study (D. Sierra-Porta, A. M. S. Borin Jr.).
**Work in progress**: the code and results here are reproducible, but the write-up is
still being written and reviewed. Comments, checks and extensions are welcome.

## The question

Ordinal (Bandt–Pompe) analysis assumes a series without ties, yet the records the
complexity–entropy plane is usually applied to are quantized: integer samples, particle
or photon counts, digitized signals. Ties are then unavoidable, and the common fix is to
break them with an arbitrary rule, which imposes an order the data do not contain. This
study asks what happens if we **keep the ties** instead, work on the generalized-ordinal-pattern
(weak-order) alphabet, and treat the quantization step as the variable of interest:

1. How does the finite-sample bias of the plane coordinates under overlapping windows behave
   on the tie alphabet?
2. How does the quantization step relocate a series in the plane, and can that shift be corrected?
3. Which null hypothesis is valid for testing a quantized series, once ties are present?

The scripts below are the numerical investigation behind these questions. They are
self-checking: each reproduces a specific result and prints its numbers or writes its figure.

## Requirements
Python 3.10+ with the packages in `requirements.txt` (`pip install -r requirements.txt`).

## Scripts

| script | what it investigates |
|---|---|
| `gop_bias_theory.py`    | exact finite-sample overlap bias on the tie alphabet; self-test that Δ→0 recovers the tie-free limit |
| `gop_theory_results.py` | compares the exact bias to Monte Carlo (m=3,4) and traces the exact plane path vs the quantization step |
| `gop_R3_sheppard.py` / `gop_R3_figure.py` | the leading-order quantization law and its test across source densities |
| `gop_R4_groundtruth.py` | tie-alphabet coordinates for white noise, AR(1), Poisson counts, the logistic map; AR(1) estimator demo |
| `gop_ar1_exact.py`      | exact quantized-AR(1) coordinates via multivariate-normal box probabilities (transfer operator) |
| `gop_R5_null.py`        | a tie-matched permutation null: its size and power vs the naive white-noise null |
| `cohort_rr.py`          | the ideas applied to real RR-interval records from two PhysioNet databases (nsr2db, Fantasia); writes `cohort_rr.csv` |
| `figure_rr_cohort.py`   | the RR cohort figure: tie bands by lattice, policy gap vs tie fraction, tie-matched null |
| `derisk_gop.py`         | first sanity check: the overlap-bias machinery ports to the 75-symbol tie alphabet |
| `verify_derivations.py` | checks the two-term discretization laws (D and H) against the exact tie-alphabet coordinates (m=3,4,5) |
| `check_estimator.py`    | Monte-Carlo check of the quantization-corrected disequilibrium estimator D̂₀ = D̂_gap + ĉ/(m−2)! (self-contained) |

Shared helpers: `fsb_extensions.py`, `apply_diagnostics.py`, `gop_rr.py` (ordinal-estimator and tie-aware plane routines).
Cached console outputs are in `results_*.txt`, figures in `figure_*.pdf` / `figure_*.png`,
small numeric caches in `*.npy`.

Example:
```
python gop_theory_results.py
python gop_R5_null.py
python cohort_rr.py          # reads the RR annotations under RealData/ (see Data below)
python figure_rr_cohort.py
```

## Data
The RR application uses beat annotations from two open PhysioNet databases at different native
lattices:

- **Normal Sinus Rhythm RR Interval Database** (`nsr2db`): 54 records, ECG at 128 Hz → 7.8125 ms
  lattice. Stein (2003), DOI [10.13026/C2S881](https://doi.org/10.13026/C2S881).
- **Fantasia Database** (`fantasia`): ECG at 250 Hz → 4 ms lattice. Peng & Lipsitz (2003),
  DOI [10.13026/C2RG61](https://doi.org/10.13026/C2RG61); original study Iyengar et al.,
  *Am. J. Physiol.* 271(4):R1078–R1084, 1996.

Both are from PhysioNet (Goldberger et al., *Circulation* 2000). The database files live under
`RealData/`. To retrieve them from PhysioNet (whole databases, incremental and resumable):

```
wget -r -N -c -np https://physionet.org/files/fantasia/1.0.0/
wget -r -N -c -np https://physionet.org/files/nsr2db/1.0.0/
```

Landing pages: <https://physionet.org/content/fantasia/1.0.0/> and
<https://physionet.org/content/nsr2db/1.0.0/>. `cohort_rr.py` reads the `.ecg`/`.hea` beat
annotations, builds the normal-to-normal RR series, and writes the per-record metrics to
`cohort_rr.csv`.

## Status
A draft write-up exists (`manuscript_ties.pdf`, read-only; the editable source is kept on
Overleaf) but it is unfinished and under review. This repository is the reproducible companion,
not the paper.
