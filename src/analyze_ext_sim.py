"""Internal protocol-defined tests H1-H4 on the 4-qubit simulation arm
(PROTOCOL_extended_study.md). Writes data/extended/sim_scores.csv (per
configuration and bandwidth) and data/extended/sim_tests.json.
"""
import glob
import json
import os
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from qiskit.quantum_info import Statevector
from scipy.stats import spearmanr, wilcoxon
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import ext_engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "data", "extended")
ARM = os.path.join(RES, "q4")
GRID = list(np.logspace(-2, 4, 25))
SHOTS = 1024
N_Q = 4
BASE = 2.0 ** -N_Q


def psd(K):
    w, V = np.linalg.eigh((K + K.T) / 2)
    return (V * np.maximum(w, 0)) @ V.T


def fit_score(Ktr, ytr, Kte, yte):
    gs = GridSearchCV(SVC(kernel="precomputed"), {"C": GRID}, cv=5).fit(
        psd(Ktr), ytr)
    return gs.best_score_, gs.score(Kte, yte)


def kta(K, y):
    s = np.where(y > 0, 1.0, -1.0)
    n = len(s)
    H = np.eye(n) - 1.0 / n
    Kc, Yc = H @ K @ H, H @ np.outer(s, s) @ H
    return float(np.sum(Kc * Yc) / (np.linalg.norm(Kc) * np.linalg.norm(Yc)))


def exact_blocks(fmap, ds, seed, c):
    Xtr, Xte, _, _ = E.load(ds, N_Q, 80, 40, seed)
    fm = E.feature_map(fmap, N_Q)
    sv = lambda X: np.array([Statevector(fm.assign_parameters(c * x)).data
                             for x in X])
    Str, Ste = sv(Xtr), sv(Xte)
    return np.abs(Str.conj() @ Str.T) ** 2, np.abs(Ste.conj() @ Str.T) ** 2


def score_file(path):
    base = os.path.basename(path)[:-4]
    fmap, _, rest = base.split("_", 2)
    noise = rest.rsplit("_", 1)[1]
    ds, seed = rest.rsplit("_", 1)[0].rsplit("_s", 1)
    seed = int(seed)
    z = np.load(path)
    ytr, yte = z["ytr"], z["yte"]
    N = len(ytr)
    iu = np.triu_indices(N, 1)
    rng = np.random.default_rng(seed)
    rows = []
    for c in z["bandwidths"]:
        off, diag, te = z[f"off_c{c}"], z[f"diag_c{c}"], z[f"te_c{c}"]
        K = np.eye(N)
        K[iu] = off
        K.T[iu] = off
        Km = K.copy()
        np.fill_diagonal(Km, diag)
        cv_u, acc_u = fit_score(K, ytr, te, yte)
        cv_m, acc_m = fit_score(Km, ytr, te, yte)
        rec = {"fmap": fmap, "dataset": ds, "seed": seed, "noise": noise,
               "c": float(c), "cv_unit": cv_u, "acc_unit": acc_u,
               "cv_meas": cv_m, "acc_meas": acc_m,
               "kta_unit": kta(K, ytr), "kta_meas": kta(Km, ytr)}
        if noise != "ideal":
            a = off[z["swap_idx"]]
            b = z[f"swap_c{c}"]
            v = (a * (1 - a) + b * (1 - b)) / SHOTS
            rec["asym"] = float(((a - b) / np.sqrt(np.maximum(v, 1e-12))).var())
            Kex, Kte_ex = exact_blocks(fmap, ds, seed, float(c))
            lam = (off.mean() - BASE) / max(Kex[iu].mean() - BASE, 1e-9)
            pa = np.clip(lam * Kex + (1 - lam) * BASE, 0, 1)
            pt = np.clip(lam * Kte_ex + (1 - lam) * BASE, 0, 1)
            Ka = np.eye(N)
            Ka[iu] = rng.binomial(SHOTS, pa[iu]) / SHOTS
            Ka.T[iu] = Ka[iu]
            Kta = rng.binomial(SHOTS, pt) / SHOTS
            _, rec["acc_affine"] = fit_score(Ka, ytr, Kta, yte)
            rec["lambda"] = float(lam)
            rec["corr_exact"] = float(np.corrcoef(off, Kex[iu])[0, 1])
        rows.append(rec)
    return rows


def tied_mean(g, col, val):
    m = g[col].max()
    t = g[np.isclose(g[col], m, rtol=0, atol=1e-12)]
    return float(t[val].mean()), float(np.log2(t.c).mean())


def main():
    cache = os.path.join(RES, "sim_scores.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
    else:
        files = sorted(glob.glob(os.path.join(ARM, "*.npz")))
        out = Parallel(n_jobs=8)(delayed(score_file)(f) for f in files)
        df = pd.DataFrame([r for rows in out for r in rows])
        df.to_csv(cache, index=False)
    sel = []
    for (fm, ds, sd), g in df.groupby(["fmap", "dataset", "seed"]):
        ideal = g[g.noise == "ideal"]
        _, l2_ideal = tied_mean(ideal, "cv_unit", "acc_unit")
        cmax = ideal.cv_unit.max()
        c_sim = set(ideal[np.isclose(ideal.cv_unit, cmax, atol=1e-12)].c)
        for nz, h in g[g.noise != "ideal"].groupby("noise"):
            acc_du, l2_du = tied_mean(h, "cv_unit", "acc_unit")
            acc_dm, l2_dm = tied_mean(h, "cv_meas", "acc_meas")
            acc_k, l2_k = tied_mean(h, "kta_unit", "acc_unit")
            acc_sim = float(h[h.c.isin(c_sim)].acc_unit.mean())
            sel.append({"fmap": fm, "dataset": ds, "seed": sd, "noise": nz,
                        "shift_unit": l2_du - l2_ideal,
                        "shift_meas": l2_dm - l2_ideal,
                        "shift_kta": l2_k - l2_ideal,
                        "acc_sim_cv": acc_sim, "acc_dev_cv_unit": acc_du,
                        "acc_dev_cv_meas": acc_dm, "acc_kta": acc_k,
                        "acc_oracle": float(h.acc_unit.max()),
                        "asym": float(h.asym.mean()),
                        "excess_loss": float((h.acc_affine - h.acc_unit).mean()),
                        "corr_exact": float(h.corr_exact.mean())})
    sel = pd.DataFrame(sel)
    sel.to_csv(os.path.join(RES, "sim_selection.csv"), index=False)
    cl = sel.groupby(["dataset", "seed"])
    res = {"n_configs": int(len(df) // 6), "n_clusters": int(cl.ngroups)}

    s = cl.shift_unit.mean()
    res["H1_drift"] = {"mean_log2_shift": float(s.mean()),
                       "p": float(wilcoxon(s[s != 0]).pvalue),
                       "clusters_up": int((s > 0).sum()),
                       "clusters_down": int((s < 0).sum()),
                       "by_noise": {k: float(v) for k, v in
                                    sel.groupby("noise").shift_unit.mean().items()}}
    d2 = cl.apply(lambda g: (g.shift_unit.abs() - g.shift_meas.abs()).mean())
    res["H2_diag_reduces_drift"] = {"mean_reduction": float(d2.mean()),
                                    "p": float(wilcoxon(d2[d2 != 0]).pvalue),
                                    "mean_abs_shift_unit": float(sel.shift_unit.abs().mean()),
                                    "mean_abs_shift_meas": float(sel.shift_meas.abs().mean())}
    rho = spearmanr(sel.asym, sel.excess_loss).correlation
    rng = np.random.default_rng(0)
    keys = list(cl.groups)
    idx = {k: v for k, v in cl.indices.items()}
    null = []
    for _ in range(2000):
        perm = rng.permutation(len(keys))
        a = np.empty(len(sel))
        for k, pk in zip(keys, perm):
            src, dst = idx[keys[pk]], idx[k]
            m = min(len(src), len(dst))
            a[dst[:m]] = sel.asym.values[src[:m]]
        null.append(spearmanr(a, sel.excess_loss).correlation)
    res["H3_asym_vs_excess_loss"] = {
        "spearman": float(rho),
        "p_cluster_perm": float(np.mean(np.abs(null) >= abs(rho))),
        "by_noise": {k: {"asym": float(g.asym.mean()),
                         "excess_loss": float(g.excess_loss.mean())}
                     for k, g in sel.groupby("noise")}}
    d4 = cl.apply(lambda g: (g.acc_dev_cv_meas - g.acc_sim_cv).mean())
    boot = [rng.choice(d4.values, len(d4)).mean() for _ in range(5000)]
    res["H4_protocol"] = {"mean_diff": float(d4.mean()),
                          "ci95": [float(np.percentile(boot, 2.5)),
                                   float(np.percentile(boot, 97.5))],
                          "p_noninferior_margin_1pt": float(
                              wilcoxon(d4 + 0.01, alternative="greater").pvalue),
                          "p_superiority": float(wilcoxon(d4[d4 != 0]).pvalue)}
    ps = {"H1": res["H1_drift"]["p"], "H2": res["H2_diag_reduces_drift"]["p"],
          "H3": res["H3_asym_vs_excess_loss"]["p_cluster_perm"],
          "H4": res["H4_protocol"]["p_noninferior_margin_1pt"]}
    order = sorted(ps, key=ps.get)
    run = 0.0
    res["holm"] = {}
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k]))
        res["holm"][k] = run
    res["rule_means"] = {k: float(sel[k].mean()) for k in
                         ["acc_oracle", "acc_sim_cv", "acc_dev_cv_unit",
                          "acc_dev_cv_meas", "acc_kta"]}
    res["rule_means_by_noise"] = sel.groupby("noise")[
        ["acc_sim_cv", "acc_dev_cv_unit", "acc_dev_cv_meas", "acc_kta"]
    ].mean().round(4).to_dict(orient="index")
    json.dump(res, open(os.path.join(RES, "sim_tests.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))




def exploratory():
    """Analyses outside the internal protocol plan; reported as exploratory."""
    sel = pd.read_csv(os.path.join(RES, "sim_selection.csv"))
    df = pd.read_csv(os.path.join(RES, "sim_scores.csv"))
    out = {"within_noise_spearman_cluster_means": {}}
    for n, g in sel.groupby("noise"):
        m = g.groupby(["dataset", "seed"])[["asym", "excess_loss"]].mean()
        r = spearmanr(m.asym, m.excess_loss)
        out["within_noise_spearman_cluster_means"][n] = {
            "rho": float(r.correlation), "p": float(r.pvalue), "n": int(len(m))}
    cl = sel.groupby(["dataset", "seed"]).apply(
        lambda g: (g.acc_dev_cv_unit - g.acc_sim_cv).mean())
    out["dev_cv_unit_minus_sim_cv"] = {
        "mean": float(cl.mean()), "p": float(wilcoxon(cl[cl != 0]).pvalue),
        "by_noise": {k: float(v) for k, v in (sel.acc_dev_cv_unit - sel.acc_sim_cv)
                     .groupby(sel.noise).mean().items()}}
    nz = df[df.noise != "ideal"]
    out["measured_minus_unit_diag_acc"] = {
        "mean": float((nz.acc_meas - nz.acc_unit).mean()),
        "frac_better": float((nz.acc_meas > nz.acc_unit).mean()),
        "frac_worse": float((nz.acc_meas < nz.acc_unit).mean())}
    ds_shift = sel.groupby("dataset").shift_unit.mean()
    ds_h4 = sel.groupby("dataset").apply(
        lambda g: (g.acc_dev_cv_meas - g.acc_sim_cv).mean())
    out["dataset_level"] = {
        "h1_shift_by_dataset": {k: float(v) for k, v in ds_shift.items()},
        "h1_positive_datasets": int((ds_shift > 0).sum()),
        "h1_wilcoxon_p": float(wilcoxon(ds_shift).pvalue),
        "h4_diff_by_dataset": {k: float(v) for k, v in ds_h4.items()},
        "h4_datasets_below_margin": int((ds_h4 < -0.01).sum())}
    cl1 = sel.groupby(["dataset", "seed"]).shift_unit.mean()
    from scipy.stats import binomtest
    out["h1_sign_test_p"] = float(binomtest(int((cl1 > 0).sum()),
                                            int((cl1 != 0).sum())).pvalue)
    out["drift_by_fmap"] = {k: float(v) for k, v in
                            sel.groupby("fmap").shift_unit.mean().items()}
    het = {}
    for f in glob.glob(os.path.join(ARM, "*.npz")):
        noise = os.path.basename(f)[:-4].rsplit("_", 1)[1]
        if noise == "ideal":
            continue
        z = np.load(f)
        for c in z["bandwidths"]:
            d = z[f"diag_c{c}"]
            m = d.mean()
            het.setdefault(noise, []).append(d.var() / max(m * (1 - m) / SHOTS,
                                                           1e-12))
    out["diag_heterogeneity_median"] = {k: float(np.median(v))
                                        for k, v in het.items()}
    json.dump(out, open(os.path.join(RES, "sim_exploratory.json"), "w"),
              indent=1)
    return out


if __name__ == "__main__":
    main()
    print(json.dumps(exploratory(), indent=1))
