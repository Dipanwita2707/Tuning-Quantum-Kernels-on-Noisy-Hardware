"""Figures for the extended (journal) study. Run after analyze_ext_sim.py
and analyze_ext.py."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "data", "extended")
FIG = os.path.join(HERE, "..", "figures", "extended")

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "savefig.bbox": "tight", "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": 0.5, "lines.linewidth": 1.4,
})
NOISE = [("dep", "Depol.", "#2a78d6", "o"),
         ("device", "Device model", "#1baf7a", "D"),
         ("cohweak", "Weak coh.", "#eda100", "^"),
         ("coh", "Strong coh.", "#e34948", "s")]


def cluster_ci(values, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    boot = [rng.choice(v, len(v)).mean() for _ in range(B)]
    return v.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def fig_drift_rules():
    sel = pd.read_csv(os.path.join(RES, "sim_selection.csv"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.2),
                               gridspec_kw={"width_ratios": [1, 1.25]})
    for k, (col, lab, colr, mk) in enumerate(
            (("shift_unit", "diagonal fixed at 1", "#eb6834", "o"),
             ("shift_meas", "measured diagonal", "#2a78d6", "s"))):
        for i, (n, nl, _, _) in enumerate(NOISE):
            g = sel[sel.noise == n].groupby(["dataset", "seed"])[col].mean()
            m, lo, hi = cluster_ci(g.values)
            a.errorbar(m, i + (k - 0.5) * 0.25, xerr=[[m - lo], [hi - m]],
                       color=colr, marker=mk, ms=4, capsize=2, lw=1.1,
                       label=lab if i == 0 else None)
    a.axvline(0, color="#52514e", lw=0.7)
    a.set_yticks(range(len(NOISE)))
    a.set_yticklabels([x[1] for x in NOISE])
    a.invert_yaxis()
    a.set_xlabel(r"shift of CV-selected $\log_2 c$ (octaves)")
    a.set_title("(a) bandwidth drift, 95% CI over 64 clusters", loc="left",
                pad=18)
    a.legend(frameon=False, fontsize=6, loc="lower center",
             bbox_to_anchor=(0.5, 0.99), ncol=2, borderaxespad=0)

    rules = [("acc_oracle", "oracle (upper bound)", "#c3c2b7"),
             ("acc_sim_cv", "CV on noiseless sim.", "#2a78d6"),
             ("acc_dev_cv_unit", "CV on device", "#1baf7a"),
             ("acc_dev_cv_meas", "CV on device, meas. diag.", "#4a3aa7"),
             ("acc_kta", "max alignment", "#eb6834")]
    w = 0.16
    for k, (col, lab, colr) in enumerate(rules):
        ms, es = [], []
        for n, _, _, _ in NOISE:
            g = sel[sel.noise == n].groupby(["dataset", "seed"])[col].mean()
            ms.append(g.mean())
            es.append(g.std() / np.sqrt(len(g)))
        x = np.arange(len(NOISE)) + (k - 2) * w
        b.bar(x, ms, w - 0.02, yerr=es, color=colr, label=lab, capsize=1.2,
              error_kw={"lw": 0.6})
    b.set_xticks(np.arange(len(NOISE)))
    b.set_xticklabels([x[1] for x in NOISE])
    b.set_ylim(0.84, 0.96)
    b.set_ylabel("test accuracy")
    b.set_title("(b) selection rules", loc="left", pad=26)
    b.legend(frameon=False, fontsize=5.6, ncol=3, loc="lower center",
             bbox_to_anchor=(0.5, 0.99), borderaxespad=0)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(os.path.join(FIG, "drift_rules.pdf"))
    plt.close(fig)


def fig_asymmetry():
    sel = pd.read_csv(os.path.join(RES, "sim_selection.csv"))
    S = json.load(open(os.path.join(RES, "summary_ext.json")))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.2))
    for n, nl, colr, mk in NOISE:
        m = sel[sel.noise == n].groupby(["dataset", "seed"])[
            ["asym", "excess_loss"]].mean()
        a.scatter(m.asym, 100 * m.excess_loss, s=9, color=colr, marker=mk,
                  alpha=0.75, lw=0, label=nl)
    a.axvline(1, color="#52514e", lw=0.7, ls="--")
    a.axhline(0, color="#52514e", lw=0.7)
    a.set_xscale("log")
    a.set_xlabel("asymmetry index (large-count reference: 1)")
    a.set_ylabel("excess loss (pts)")
    a.set_title("(a) simulation, cluster means", loc="left")
    a.legend(frameon=False, fontsize=6, loc="upper left")

    hw = pd.DataFrame(S["hardware_runs"].values())
    devs = []
    nulls = {}
    for dev, g in hw.groupby("device"):
        devs.append((dev.replace("ibm_", "IBM "), g.asym_index.values, None))
        nulls[dev.replace("ibm_", "IBM ")] = g.asym_null_975.values
    for k, v in S["vendor_diagnostics"].items():
        name = "IQM " + k.split("_")[1].capitalize()
        devs.append((name, [v["asym_index_swap"]],
                     v["asym_index_repeat_control"]))
    for i, (name, vals, ctrl) in enumerate(devs):
        if name in nulls:
            b.scatter(np.full(len(nulls[name]), i + 0.18), nulls[name],
                      marker="_", s=40, color="#c3c2b7", zorder=2,
                      label="pooled null, 97.5%" if i == 0 else None)
        colr = "#4a3aa7" if name.startswith("IBM") else "#1baf7a"
        b.scatter(np.full(len(vals), i), vals, color=colr, s=14, zorder=3,
                  label=("IBM blocks" if name.startswith("IBM") else
                         "IQM (forward vs swapped)") if i in (0, len(devs) - 1)
                  else None)
        if ctrl is not None:
            b.scatter([i], [ctrl], marker="_", s=120, color="#52514e",
                      zorder=4, label="IQM repeat control"
                      if i == len(devs) - 1 else None)
    b.axhline(1, color="#52514e", lw=0.7, ls="--")
    b.set_xticks(range(len(devs)))
    b.set_xticklabels([d[0] for d in devs], rotation=20, ha="right")
    b.set_ylabel("asymmetry index")
    b.set_title("(b) hardware", loc="left")
    b.legend(frameon=False, fontsize=5.6, loc="upper right")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(os.path.join(FIG, "asymmetry.pdf"))
    plt.close(fig)


def fig_hardware_diag():
    S = json.load(open(os.path.join(RES, "summary_ext.json")))
    nst = json.load(open(os.path.join(RES, "hardware_nested_stats.json")))
    hw = pd.DataFrame(nst["rows"]).sort_values(["device", "c"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.1))
    short = {"ibm_fez": "fez", "ibm_kingston": "king.", "ibm_marrakesh": "marr."}
    labels = [f"{short[d]}\n1/{int(round(1 / c))}"
              for d, c in zip(hw.device, hw.c)]
    x = np.arange(len(hw))
    for k, (col, lab, colr, mk) in enumerate(
            (("nested_exact", "exact kernel", "#52514e", "_"),
             ("nested_unit", "diagonal fixed at 1", "#eb6834", "o"),
             ("nested_meas", "measured diagonal", "#2a78d6", "s"),
             ("nested_mit", "per-point mitigated", "#4a3aa7", "^"))):
        a.scatter(x + (k - 1.5) * 0.12, hw[col], s=60 if mk == "_" else 14,
                  color=colr, marker=mk, label=lab, zorder=3,
                  linewidths=1.6 if mk == "_" else 0)
    a.set_xticks(x)
    a.set_xticklabels(labels, fontsize=6)
    a.set_ylim(0.4, 1.0)
    a.set_ylabel("transductive CV accuracy")
    a.set_title("(a) IBM, digits, 32 points", loc="left", pad=24)
    a.legend(frameon=False, fontsize=5.6, ncol=2, loc="lower center",
             bbox_to_anchor=(0.5, 0.99), borderaxespad=0)

    allhw = pd.DataFrame(S["hardware_runs"].values())
    pal = {"ibm_fez": ("#2a78d6", "o"), "ibm_kingston": ("#eb6834", "s"),
           "ibm_marrakesh": ("#1baf7a", "D")}
    for dev, g in allhw.groupby("device"):
        colr, mk = pal[dev]
        b.scatter(g.lambda_offdiag_slope, g.lambda_diag_affine, s=16, color=colr,
                  marker=mk, label=dev.replace("ibm_", "IBM "), alpha=0.85)
    for (k, v), colr in zip(S["vendor_diagnostics"].items(),
                            ("#4a3aa7", "#e34948")):
        b.scatter([v["lambda_offdiag_slope"]], [v["lambda_diag_affine"]],
                  marker="*", s=70, color=colr,
                  label="IQM " + k.split("_")[1].capitalize())
    b.plot([0, 1], [0, 1], color="#52514e", lw=0.7, ls="--")
    b.set_xlim(0, 1.05), b.set_ylim(0, 1.05)
    b.set_xlabel("slope of hardware on exact entries")
    b.set_ylabel(r"diagonal-based $\hat\lambda_{\mathrm{d}}$")
    b.set_title("(b) global-affine parameter diagnostics", loc="left")
    b.legend(frameon=False, fontsize=5.6, loc="lower right")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(os.path.join(FIG, "hardware_diag.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_drift_rules()
    if os.path.exists(os.path.join(RES, "summary_ext.json")):
        S = json.load(open(os.path.join(RES, "summary_ext.json")))
        if S.get("hardware_runs"):
            fig_asymmetry()
            fig_hardware_diag()
    print("figures in", FIG)
