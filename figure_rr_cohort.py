"""Cohort RR figure across two PhysioNet databases (nsr2db, Fantasia) at different lattices."""
import numpy as np, csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gop_rr as R6
from cohort_rr import load_rr, record_path, pval_two

HERE = Path(__file__).resolve().parent
W = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 9,
    "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
    "ytick.right": True, "legend.fontsize": 7.0, "pdf.fonttype": 42})
def sm(ax): ax.minorticks_on(); ax.tick_params(which="both", direction="in", top=True, right=True)

# ---- cohort table ----
rows = list(csv.DictReader(open(HERE / "cohort_rr.csv")))
tie = np.array([float(r["tie"]) for r in rows]) * 100
gap = np.array([float(r["gap"]) for r in rows])
isF = np.array([r["db"] == "fantasia" for r in rows])
corr = np.corrcoef(tie, gap)[0, 1]

# ---- showcase record (fantasia, 250 Hz, finer lattice) ----
SHOW, FS = "f1o05", 250
rr = load_rr(record_path("fantasia", SHOW), FS)
g = R6.gop_coords(rr)
p2, T0, Tn = pval_two(rr)
ms = 1000.0 / FS

fig, ax = plt.subplots(2, 2, figsize=(7.6, 5.6), layout="constrained")

# (a) RR trace of showcase
axa = ax[0, 0]
axa.plot(np.arange(len(rr)) , rr * ms, color=W[0], lw=0.5)
axa.set_xlabel("beat number"); axa.set_ylabel("RR interval (ms)")
axa.set_title(f"(a) RR example ({SHOW}, 250 Hz, tie {100*g['tie']:.0f}%, null $p={p2:.3f}$)", loc="left")
sm(axa)

# (b) tie bands by database (the lattice sets the band)
axb = ax[0, 1]
for k, (name, mk, lab) in enumerate([(False, "o", r"nsr2db (128 Hz, $\Delta{=}7.8$ ms)"),
                                     (True, "s", r"fantasia (250 Hz, $\Delta{=}4.0$ ms)")]):
    sel = isF == name
    xj = k + (np.random.default_rng(k).random(sel.sum()) - 0.5) * 0.3
    axb.scatter(xj, tie[sel], s=16, color=W[k], marker=mk, edgecolor="k", lw=0.3, label=lab, zorder=3)
    axb.plot([k - 0.25, k + 0.25], [np.median(tie[sel])] * 2, color="k", lw=1.6, zorder=4)
axb.set_xticks([0, 1]); axb.set_xticklabels(["nsr2db", "fantasia"])
axb.set_ylabel("tied windows (%)")
axb.set_title(r"(b) Coarser lattice $\to$ more ties", loc="left")
axb.legend(loc="upper right", fontsize=6.3); sm(axb)

# (c) cohort: policy gap vs tie fraction
axc = ax[1, 0]
for k, (name, mk) in enumerate([(False, "o"), (True, "s")]):
    sel = isF == name
    axc.scatter(tie[sel], gap[sel], s=16, color=W[k], marker=mk, edgecolor="k", lw=0.3, zorder=3)
b1, b0 = np.polyfit(tie, gap, 1); xx = np.linspace(tie.min(), tie.max(), 50)
axc.plot(xx, b0 + b1 * xx, "k--", lw=0.9, zorder=2)
axc.axhline(0, color=W[2], lw=1.3, label="GOP (tie-aware): no policy, zero gap")
axc.set_xlabel("tied windows (%)"); axc.set_ylabel(r"$H_{\rm index}-H_{\rm jitter}$ (tie-broken)")
axc.set_title(f"(c) Policy gap grows with ties (corr $={corr:.2f}$, 61 records)", loc="left")
axc.legend(loc="lower left"); sm(axc)

# (d) GOP plane: showcase vs its tie-matched null (two-sided)
axd = ax[1, 1]
rng = np.random.default_rng(3)
xs = rr[:4000] if len(rr) > 4000 else rr
Hn, Cn = [], []
for _ in range(400):
    co = R6.gop_coords(rng.permutation(xs)); Hn.append(co["H"]); Cn.append(co["C_JS"])
gg = R6.gop_coords(xs)
axd.scatter(Hn, Cn, s=8, color="0.7", label="tie-matched null", zorder=1)
axd.scatter([gg["H"]], [gg["C_JS"]], s=80, color=W[1], edgecolor="k", lw=0.6,
            marker="*", zorder=3, label=f"{SHOW} (observed)")
axd.set_xlabel(r"normalized entropy $H$ (base $\log 75$)"); axd.set_ylabel(r"$C_{\rm JS}$")
axd.set_title("(d) Observed sits off (flatter than) its null", loc="left")
axd.legend(loc="upper right", fontsize=6.6); sm(axd)

fig.savefig(HERE / "figure_R6.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
fig.savefig(HERE / "figure_R6.pdf", bbox_inches="tight", pad_inches=0.05)
print(f"showcase {SHOW}: H={g['H']:.4f} D={g['D']:.5f} C={g['C_JS']:.5f} tie={100*g['tie']:.0f}% p2={p2:.3f} (T0<null={T0<Tn})")
print(f"cohort corr={corr:.3f}; saved figure_R6.png/pdf")
