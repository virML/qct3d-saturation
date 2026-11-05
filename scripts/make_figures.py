"""Manuscript figures for 'Quantum or just quantised?', drawn from the CSVs the tables quote.

Springer artwork rules applied: Arial-metric sans (Liberation Sans) at >= 8 pt, no titles or
caption text inside the artwork, markers in addition to colour, width <= 174 mm, vector EPS/PDF
for production and 600-dpi PNG for the review manuscript.

usage:  python make_figs.py <csv_dir> <out_dir>
"""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np, pandas as pd

CSV, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
W = 6.3   # inches = 160 mm, inside the 174 mm full-width limit
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "grid.linewidth": 0.5, "lines.linewidth": 1.4,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType outlines, keep text editable
})
QC, CC, LC, RED = "#1f4e99", "#c9721a", "#6b6b6b", "#b03030"
BB = {"B1_swinunetr_suprem": "B1 SwinUNETR", "B2_segresnet_suprem": "B2 SegResNet",
      "B3_resnet18_medicalnet": "B3 ResNet-18"}
FAM = {"Q1_sel": "Q1 strongly-entangling", "Q2_reupload": "Q2 re-uploading",
       "Q4_ttn": "Q4 tree tensor network"}
MK = {"B1_swinunetr_suprem": "o", "B2_segresnet_suprem": "s", "B3_resnet18_medicalnet": "^"}

t1 = pd.read_csv(f"{CSV}/table1_conditioning.csv")
t3 = pd.read_csv(f"{CSV}/table3_label_efficiency.csv")
t4 = pd.read_csv(f"{CSV}/table4_transfer.csv")
sA = pd.read_csv(f"{CSV}/stats_A_quantum_vs_classical.csv")
sB = pd.read_csv(f"{CSV}/stats_B_placement.csv")
for df in (t1,):
    df["encoder_norm"] = df["encoder_norm"].astype(str).str.lower().eq("true")
for df in (sB,):
    df["sig_bh"] = df["sig_bh"].astype(str).str.lower().eq("true")

def tint(hexcol, a):
    """Solid colour equal to hexcol at opacity a over white -- EPS has no transparency."""
    h = hexcol.lstrip("#"); rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(255 - a * (255 - v)):02x}" for v in rgb)

GRID = dict(color="#e2e2e2", linewidth=0.5)

def panel(ax, letter, dx=-0.16, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left")

def save(fig, n):
    for ext, kw in (("eps", {}), ("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(f"{OUT}/Fig{n}.{ext}", **kw)
    plt.close(fig)
    print(f"  Fig{n}: eps, pdf, png@600dpi")

# ------------------------------------------------------------------------- Fig 1
fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.6), sharey=True)
fig.subplots_adjust(wspace=0.12)
q = t1[t1.arm == "quantum"]
for bb, g in q.groupby("backbone"):
    u, c = g[~g.encoder_norm], g[g.encoder_norm]
    a.scatter(u.pretanh_median, u.auroc_mean, s=28, marker=MK[bb], c=QC, edgecolors="white",
              linewidths=0.5, zorder=3)
    a.scatter(c.pretanh_median, c.auroc_mean, s=28, marker=MK[bb], facecolors="white",
              edgecolors=QC, linewidths=1.0, zorder=3)
a.axvline(10, color=RED, ls="--", lw=1.0, zorder=1)
a.text(13, 0.888, "|z| = 10: gradient\nunderflows to zero", color=RED, fontsize=8, va="top")
a.set_xscale("log"); a.set_xlim(0.4, 2000); a.set_ylim(0.55, 0.90)
a.set_xlabel("median |z| entering tanh (log scale)"); a.set_ylabel("AUROC")
a.grid(axis="y", **GRID); a.set_axisbelow(True)
# shape = backbone (drawn neutral), fill = conditioning (drawn with a shape no backbone uses)
h = [Line2D([], [], ls="", marker=MK[k], color="#555555", ms=5.5, label=v) for k, v in BB.items()]
h += [Line2D([], [], ls="", marker="h", color=QC, ms=6.5, label="unconditioned (filled)"),
      Line2D([], [], ls="", marker="h", mfc="white", mec=QC, mew=1.1, ms=6.5, label="conditioned (open)")]
leg = a.legend(handles=h, loc="lower left", handletextpad=0.3, borderaxespad=0.3, ncol=1,
               frameon=True, framealpha=1.0, facecolor="white", edgecolor="none")
leg.set_zorder(5)   # opaque box so the |z| = 10 line does not run through the legend text
panel(a, "a")
m = q[~q.encoder_norm].merge(q[q.encoder_norm], on=["backbone", "family"], suffixes=("_u", "_c"))
for r in m.itertuples():
    b.plot([0, 1], [r.auroc_mean_u, r.auroc_mean_c], "-", color="#a9b8d3", lw=1.0, zorder=1)
    b.plot(0, r.auroc_mean_u, MK[r.backbone], color=QC, ms=4.5, zorder=3)
    b.plot(1, r.auroc_mean_c, MK[r.backbone], color=QC, ms=4.5, mfc="white", mew=1.1, zorder=3)
b.set_xlim(-0.35, 1.35); b.set_xticks([0, 1]); b.set_xticklabels(["unconditioned", "conditioned"])
b.grid(axis="y", **GRID); b.set_axisbelow(True)
panel(b, "b", dx=-0.06)
save(fig, 1)

# ------------------------------------------------------------------------- Fig 2
fig, axes = plt.subplots(1, 2, figsize=(W, 2.75), sharey=True)
fig.subplots_adjust(bottom=0.24, wspace=0.10)
fams = ["Q1_sel", "Q2_reupload", "Q4_ttn"]
x = np.arange(3); wdt = 0.26
HATCH = ["", "////", ".."]
handles = []
for ax, nm, ttl in zip(axes, [False, True], ["unconditioned", "conditioned"]):
    d = t1[(t1.arm == "quantum") & (t1.encoder_norm == nm)]
    for j, f in enumerate(fams):
        s_ = d[d.family == f].set_index("backbone").reindex(list(BB))
        bars = ax.bar(x + (j - 1) * wdt, s_.auroc_mean, wdt * 0.92, yerr=s_.auroc_sd,
                      color=[QC, "#4e79b8", "#a8c2e4"][j], hatch=HATCH[j], edgecolor="white",
                      linewidth=0.4, error_kw=dict(lw=0.8, capsize=2, ecolor="#333333"), zorder=3)
        if nm is False:
            handles.append(bars)
    sp = d.groupby("backbone").auroc_mean.agg(lambda t: t.max() - t.min()).mean()
    ax.set_title(ttl, fontsize=9, pad=4)
    ax.text(0.5, 0.97, f"mean spread {sp:.3f}", transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color="#444444")
    ax.set_xticks(x); ax.set_xticklabels([v.split()[0] for v in BB.values()])
    ax.grid(axis="y", **GRID); ax.set_axisbelow(True)
axes[0].set_ylabel("AUROC"); axes[0].set_ylim(0.5, 0.92)
panel(axes[0], "a", dy=1.06); panel(axes[1], "b", dx=-0.06, dy=1.06)
fig.legend(handles, [FAM[f] for f in fams], loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.01), handletextpad=0.5, columnspacing=1.6)
save(fig, 2)

# ------------------------------------------------------------------------- Fig 3
fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.45})
d = sA.sort_values("delta"); y = np.arange(len(d))
a.axvspan(-0.02, 0.02, color="#e6ecf5", zorder=0)
a.errorbar(d.delta, y, xerr=[d.delta - d.lo, d.hi - d.delta], fmt="o", ms=3.8, color=QC,
           ecolor="#8ea6cc", elinewidth=1.1, capsize=2, zorder=3)
a.axvline(0, color="#444444", lw=0.9, zorder=2)
a.set_yticks(y); a.set_yticklabels(d.cell)
a.set_ylim(-0.6, len(d) - 0.4)
a.set_xlabel("quantum − matched classical (AUROC)")
a.set_title("Family A", fontsize=9, pad=4)
a.grid(axis="x", **GRID); a.set_axisbelow(True)
panel(a, "a", dx=-0.32)
d2 = sB.sort_values("delta"); y2 = np.arange(len(d2))
b.errorbar(d2.delta, y2, xerr=[d2.delta - d2.lo, d2.hi - d2.delta], fmt="none",
           ecolor="#8ea6cc", elinewidth=1.1, capsize=2, zorder=3)
for xi, yi, sig in zip(d2.delta, y2, d2.sig_bh):
    b.plot(xi, yi, "D" if sig else "o", ms=5 if sig else 4.4, color=RED if sig else QC, zorder=4)
    if sig:
        b.annotate("q = 0.013", (xi, yi), xytext=(8, 7), textcoords="offset points", color=RED,
                   fontsize=8, va="bottom")
b.axvline(0, color="#444444", lw=0.9, zorder=2)
b.set_yticks(y2); b.set_yticklabels(d2.cell)
b.set_xlim(-0.037, 0.026); b.set_ylim(-0.6, len(d2) - 0.35)
b.set_xlabel("spatial − pooled (AUROC)")
b.set_title("Family B", fontsize=9, pad=4)
b.grid(axis="x", **GRID); b.set_axisbelow(True)
panel(b, "b", dx=-0.45)
save(fig, 3)

# ------------------------------------------------------------------------- Fig 4 / Fig 5
def curves(target, n):
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.35), sharey=True)
    for ax, bb in zip(axes, BB):
        d = t3[(t3.backbone == bb) & (t3.target == target)]
        for arm, col, lab, ls, mk in (("quantum", QC, "quantum", "-", "o"),
                                      ("classical", CC, "classical (matched)", "-", "s"),
                                      ("linear_probe", LC, "linear probe", "--", "^")):
            s = d[d.arm == arm].sort_values("k")
            se = s.auroc_sd / np.sqrt(s.n_draws)
            ax.plot(s.k, s.auroc_mean, ls, color=col, marker=mk, ms=3.6, label=lab, zorder=3)
            ax.fill_between(s.k, s.auroc_mean - se, s.auroc_mean + se, color=tint(col, 0.18), lw=0, zorder=2)
        ax.set_xscale("log"); ax.set_xticks([25, 50, 100, 200, 400, 900])
        ax.set_xticklabels(["25", "50", "100", "200", "400", "900"])
        ax.minorticks_off()
        ax.set_xlabel("local labels k")
        ax.axhline(0.5, color="#999999", lw=0.7, ls=":")
        ax.set_title(BB[bb], fontsize=9, pad=4)
        ax.grid(axis="y", **GRID); ax.set_axisbelow(True)
    axes[0].set_ylabel("AUROC"); axes[0].set_ylim(0.46, 0.80)
    axes[0].legend(loc="upper left", handletextpad=0.5, borderaxespad=0.2)
    save(fig, n)
curves("abnormal_any", 4)
curves("hemorrhage", 5)

# ------------------------------------------------------------------------- Fig 6
fig, axes = plt.subplots(1, 3, figsize=(W, 2.35), sharey=True)
for ax, bb in zip(axes, BB):
    d = t4[t4.backbone == bb]
    ad = d[d["mode"] != "zero"]
    ks = sorted(ad.k.unique())
    for arm, col, lab, mk in (("quantum", QC, "best adapted quantum", "o"),
                              ("classical", CC, "best adapted classical", "s")):
        ys = [ad[(ad.k == k) & (ad.arm == arm)].auroc_mean.max() for k in ks]
        ax.plot(ks, ys, "-", color=col, marker=mk, ms=3.6, label=lab, zorder=3)
    lp = d[(d.arm == "linear_probe") & (d.k > 0)].drop_duplicates("k").sort_values("k")
    ax.plot(lp.k, lp.auroc_mean, "--", color=LC, marker="^", ms=3.6, label="local labels only", zorder=3)
    zc = d[(d["mode"] == "zero") & (d.arm == "classical")].auroc_mean.iloc[0]
    ax.axhline(zc, color=CC, lw=1.1, ls=":", zorder=1,
               label="classical zero-shot" if bb == list(BB)[0] else None)
    ax.set_xscale("log"); ax.set_xticks([10, 25, 50, 100, 200, 400, 900])
    ax.set_xticklabels(["10", "", "50", "", "200", "", "900"])   # every budget ticked, alternate ones labelled
    ax.minorticks_off()
    ax.set_xlabel("local labels k")
    ax.axhline(0.5, color="#999999", lw=0.7, ls=":")
    ax.set_title(BB[bb], fontsize=9, pad=4)
    ax.grid(axis="y", **GRID); ax.set_axisbelow(True)
axes[0].set_ylabel("AUROC (hemorrhage)"); axes[0].set_ylim(0.44, 0.80)
axes[0].legend(loc="upper left", handletextpad=0.5, borderaxespad=0.2)
save(fig, 6)
print("all figures written to", OUT)
