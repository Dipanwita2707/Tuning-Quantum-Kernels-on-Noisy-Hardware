"""Extended-study analysis (PROTOCOL_extended_study.md). Writes
data/extended/summary_ext.json. Sections are filled as data arrive.
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "data", "extended")


def zscores(a, b, s):
    v = (a * (1 - a) + b * (1 - b)) / s
    return (a - b) / np.sqrt(np.maximum(v, 1e-12))


def asymmetry_null(fwd, swapped, shots=512, draws=10000, seed=0):
    """Pointwise plug-in null for equal forward/reversed probabilities.

    With equal shot counts the pooled probability is the constrained MLE.
    Recompute the observed statistic, including its estimated denominator,
    in each draw. This null assumes independent binomial sampling and does
    not remove execution-order drift or adjust across hardware blocks.
    """
    pooled = (np.asarray(fwd) + np.asarray(swapped)) / 2
    rng = np.random.default_rng(seed)
    a = rng.binomial(shots, pooled, size=(draws, len(pooled))) / shots
    b = rng.binomial(shots, pooled, size=(draws, len(pooled))) / shots
    null = zscores(a, b, shots).var(axis=1)
    observed = zscores(fwd, swapped, shots).var()
    return {
        "asym_null_mean": float(null.mean()),
        "asym_null_975": float(np.percentile(null, 97.5)),
        "asym_null_p": float((1 + np.count_nonzero(null >= observed)) /
                             (draws + 1)),
        "asym_null_method": "pooled_equal_probability_binomial",
        "asym_null_draws": draws,
        "asym_null_seed": seed,
    }


def refresh_hardware_nulls():
    """Recompute nulls from raw blocks without repeating unchanged SVM fits."""
    path = os.path.join(RES, "summary_ext.json")
    with open(path) as f:
        summary = json.load(f)
    for key, row in summary["hardware_runs"].items():
        run, bandwidth = key.split("|")
        raw = os.path.join(RES, "hardware", run, f"gram_{bandwidth}.npz")
        with np.load(raw) as z:
            row.update(asymmetry_null(z["off"][z["swap_idx"]], z["swap"]))
    summary["hardware_stats"] = hardware_stats(summary)
    with open(path, "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def boot_ci(fn, *arrays, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals.append(fn(*[x[idx] for x in arrays]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def vendor_diagnostics(n_q=4):
    """Asymmetry (forward vs swapped), block control (forward vs repeat),
    signal retention, and fidelity to the exact kernel per IQM chip."""
    out = {}
    for d in sorted(glob.glob(os.path.join(RES, "oq_diag", "*"))):
        p = os.path.join(d, "entries.json")
        if not os.path.exists(p):
            continue
        rows = json.load(open(p))
        shots = int(d.rsplit("_s", 1)[1])
        by = {}
        for r in rows:
            key = (min(r["a"], r["b"]), max(r["a"], r["b"]))
            by.setdefault(key, {})[r["kind"]] = r
        if not rows:
            continue
        pairs = [v for k, v in by.items() if k[0] != k[1] and
                 {"fwd", "swap", "rep"} <= set(v)]
        f = np.array([v["fwd"]["hw"] for v in pairs])
        s = np.array([v["swap"]["hw"] for v in pairs])
        r = np.array([v["rep"]["hw"] for v in pairs])
        ex = np.array([v["fwd"]["exact"] for v in pairs])
        diag = np.array([v["diag"]["hw"] for k, v in by.items()
                         if k[0] == k[1] and "diag" in v])
        z_sw, z_rep = zscores(f, s, shots), zscores(f, r, shots)
        base = 2.0 ** -n_q
        lam_off = ((np.r_[f, s, r].mean() - base) /
                   (np.r_[ex, ex, ex].mean() - base))
        lam_diag = float(np.sqrt(max(diag.mean() - base, 0) / (1 - base)))
        lam_diag_aff = float((diag.mean() - base) / (1 - base))
        lam_slope = float(np.polyfit(np.r_[ex, ex], np.r_[f, s], 1)[0])
        out[os.path.basename(d)] = {
            "pairs": int(len(pairs)), "diag_points": int(len(diag)),
            "asym_index_swap": float(z_sw.var()),
            "asym_index_repeat_control": float(z_rep.var()),
            "direction_effect": float(z_sw.var() - z_rep.var()),
            "direction_effect_ci95": boot_ci(
                lambda a, b: a.var() - b.var(), z_sw, z_rep),
            "lambda_offdiag_mean": float(lam_off),
            "lambda_from_diagonal": lam_diag,
            "lambda_diag_affine": lam_diag_aff,
            "lambda_offdiag_slope": lam_slope,
            "repeat_ci_upper": float(boot_ci(lambda a, b: a.var() - b.var(),
                                             z_sw, z_rep)[1]),
            "diag_mean": float(diag.mean()),
            "diag_heterogeneity": float(
                diag.var() / max(diag.mean() * (1 - diag.mean()) / shots,
                                 1e-12)),
            "corr_exact_fwd": float(np.corrcoef(f, ex)[0, 1]),
            "corr_exact_sym": float(np.corrcoef((f + s) / 2, ex)[0, 1]),
            "sentinel_min": float(min(x["sentinel"] for x in rows))}
    return out


def _exact_gram(task, n, c):
    import ext_engine as E
    from qiskit.quantum_info import Statevector
    import hardware_ext as H
    X, _ = H.task_data(task, n)
    fm = E.feature_map("zz", n)
    sv = np.array([Statevector(fm.assign_parameters(c * x)).data for x in X])
    return np.abs(sv.conj() @ sv.T) ** 2


def _repcv(K, y, n_splits=100):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.svm import SVC
    w, V = np.linalg.eigh((K + K.T) / 2)
    K = (V * np.maximum(w, 0)) @ V.T
    grid = np.logspace(-2, 4, 25)
    return float(np.mean([
        max(cross_val_score(SVC(kernel="precomputed", C=C), K, y,
                            cv=StratifiedKFold(5, shuffle=True,
                                               random_state=r)).mean()
            for C in grid) for r in range(n_splits)]))


def hardware_runs(n_splits=100):
    """IBM extended-protocol runs: per (device, task, day, bandwidth) the
    label-free diagnostics and CV accuracy under three diagonal treatments
    (unit, measured, per-point mitigated), against the exact kernel."""
    out = {}
    for d in sorted(glob.glob(os.path.join(RES, "hardware", "*"))):
        name = os.path.basename(d)
        dev_task = name.split("_n")[0]
        task = dev_task.split("_", 2)[2] if dev_task.count("_") >= 2 else ""
        n = int(name.split("_n")[1].split("_")[0])
        for p in sorted(glob.glob(os.path.join(d, "gram_c*.npz"))):
            c = float(p.split("gram_c")[1][:-4])
            z = np.load(p)
            off, diag, sw, y = z["off"], z["diag"], z["swap"], z["y"]
            N = len(diag)
            iu = np.triu_indices(N, 1)
            Kex = _exact_gram(task, n, c)
            ex = Kex[iu]
            shots = 512
            fwd = off[z["swap_idx"]]
            zsw = zscores(fwd, sw, shots)
            base = 2.0 ** -n
            lam_off = (off.mean() - base) / (ex.mean() - base)
            # affine model K~ = lam K + (1-lam) 2^-n: slope of hardware on
            # exact entries; the diagonal implies lam = (K_ii - b)/(1 - b)
            lam_slope = float(np.polyfit(ex, off, 1)[0])
            lam_diag_affine = float((diag.mean() - base) / (1 - base))
            null_stats = asymmetry_null(fwd, sw, shots)
            lam_i = np.sqrt(np.clip((diag - base) / (1 - base), 1e-6, None))
            K_unit = np.eye(N)
            K_unit[iu] = off
            K_unit.T[iu] = off
            K_meas = K_unit.copy()
            np.fill_diagonal(K_meas, diag)
            ll = np.outer(lam_i, lam_i)
            K_mit = (K_unit - (1 - ll) * base) / ll
            np.fill_diagonal(K_mit, 1.0)
            kb = off.mean()
            out[f"{name}|c{c}"] = {
                "device": name.split("_")[0] + "_" + name.split("_")[1],
                "task": task, "n": n, "c": c,
                "corr_exact": float(np.corrcoef(off, ex)[0, 1]),
                "lambda_offdiag": float(lam_off),
                "lambda_offdiag_slope": lam_slope,
                "lambda_diag_affine": lam_diag_affine,
                **null_stats,
                "lambda_diag_mean": float(np.sqrt(max(diag.mean() - base, 0)
                                                  / (1 - base))),
                "diag_mean": float(diag.mean()),
                "diag_heterogeneity": float(diag.var() / max(
                    diag.mean() * (1 - diag.mean()) / shots, 1e-12)),
                "asym_index": float(zsw.var()),
                "snr_hat": float(shots * off.var() / (kb * (1 - kb)) - 1),
                "acc_exact": _repcv(Kex, y, n_splits),
                "acc_unit_diag": _repcv(K_unit, y, n_splits),
                "acc_measured_diag": _repcv(K_meas, y, n_splits),
                "acc_mitigated": _repcv(K_mit, y, n_splits),
                "quantum_seconds": float(z["quantum_seconds"]),
                "finished": str(z["finished"])}
    return out


def main():
    summary = {"vendor_diagnostics": vendor_diagnostics(),
               "hardware_runs": hardware_runs()}
    summary["hardware_stats"] = hardware_stats(summary)
    json.dump(summary, open(os.path.join(RES, "summary_ext.json"), "w"),
              indent=1)
    print(json.dumps(summary, indent=1))




def hardware_stats(S):
    """Summary statistics over the IBM blocks quoted in the paper."""
    import pandas as pd
    hw = pd.DataFrame(S["hardware_runs"].values())
    d = hw.acc_measured_diag - hw.acc_unit_diag
    out = {"blocks": int(len(hw)),
           "meas_diag_better": int((d > 0).sum()),
           "meas_diag_worse": int((d < 0).sum()),
           "meas_diag_gain_mean": float(d.mean()),
           "meas_diag_gain_digits_mean": float(d[hw.task == "digits3v5"].mean()),
           "meas_diag_gain_max": float(d.max()),
           "asym_median": float(hw.asym_index.median()),
           "asym_min": float(hw.asym_index.min()),
           "asym_max": float(hw.asym_index.max()),
           "asym_gt_1p36": int((hw.asym_index > 1.36).sum()),
           "diag_affine_gt_offdiag_ratio": int((hw.lambda_diag_affine >
                                                hw.lambda_offdiag).sum()),
           "diag_affine_gt_offdiag_slope": int((hw.lambda_diag_affine >
                                                hw.lambda_offdiag_slope).sum()),
           "asym_gt_block_null975": int((hw.asym_index >
                                         hw.asym_null_975).sum()),
           "asym_null_mean_range": [float(hw.asym_null_mean.min()),
                                    float(hw.asym_null_mean.max())],
           "asym_median_by_c": {str(k): float(v) for k, v in
                                hw.groupby("c").asym_index.median().items()},
           "spearman_D_vs_diag_gain": [float(x) for x in __import__(
               "scipy.stats", fromlist=["spearmanr"]).spearmanr(
               hw.diag_heterogeneity, d)],
           "diag_gain_by_device": {k: float(v) for k, v in
                                   d.groupby(hw.device).mean().items()},
           "diag_het_median": float(hw.diag_heterogeneity.median()),
           "diag_het_max": float(hw.diag_heterogeneity.max()),
           "corr_exact_range": [float(hw.corr_exact.min()),
                                float(hw.corr_exact.max())],
           "snr_hat_range": [float(hw.snr_hat.min()), float(hw.snr_hat.max())],
           "best_c": {}}
    for (dev, task), g in hw.groupby(["device", "task"]):
        out["best_c"][f"{dev}|{task}"] = {
            col: float(g.loc[g[col].idxmax(), "c"])
            for col in ("acc_exact", "acc_unit_diag", "acc_measured_diag")}
    out["quantum_seconds_by_device"] = {
        k: float(v) for k, v in hw.groupby("device").quantum_seconds.sum().items()}
    return out


def hardware_bootstrap(B=40, n_splits=20, task="digits3v5"):
    """Parametric shot bootstrap of the measured-diagonal gain per IBM
    block: every off-diagonal and diagonal entry is redrawn from
    Binomial(512, estimate), and the gain is recomputed on the same splits."""
    out = {}
    rng = np.random.default_rng(7)
    for d in sorted(glob.glob(os.path.join(RES, "hardware", f"*_{task}_*"))):
        for p in sorted(glob.glob(os.path.join(d, "gram_c*.npz"))):
            z = np.load(p)
            off, diag, y = z["off"], z["diag"], z["y"]
            N = len(diag)
            iu = np.triu_indices(N, 1)
            gains = []
            for _ in range(B):
                o = rng.binomial(512, off) / 512
                dg = rng.binomial(512, diag) / 512
                K = np.eye(N)
                K[iu] = o
                K.T[iu] = o
                Km = K.copy()
                np.fill_diagonal(Km, dg)
                gains.append(_repcv(Km, y, n_splits) - _repcv(K, y, n_splits))
            g = np.array(gains)
            out[f"{os.path.basename(d)}|{p.split('gram_c')[1][:-4]}"] = {
                "mean": float(g.mean()), "ci95": [float(np.percentile(g, 2.5)),
                                                  float(np.percentile(g, 97.5))],
                "frac_positive": float(np.mean(g > 0))}
    json.dump(out, open(os.path.join(RES, "hardware_bootstrap.json"), "w"),
              indent=1)
    return out


def _nested_cv(K, y, n_splits=20):
    """Transductive accuracy with nested C selection (inner 4-fold CV).

    Whole-matrix eigenvalue clipping precedes the repeated outer 5-fold
    splits. Feature preprocessing is fixed by acquisition, and bandwidth
    selection is not nested. This is not a fully inductive evaluation.
    """
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.svm import SVC
    w, V = np.linalg.eigh((K + K.T) / 2)
    K = (V * np.maximum(w, 0)) @ V.T
    grid = list(np.logspace(-2, 4, 25))
    accs = []
    for r in range(n_splits):
        for tr, te in StratifiedKFold(5, shuffle=True,
                                      random_state=r).split(K, y):
            gs = GridSearchCV(SVC(kernel="precomputed"), {"C": grid},
                              cv=StratifiedKFold(4, shuffle=True,
                                                 random_state=100 + r))
            gs.fit(K[np.ix_(tr, tr)], y[tr])
            accs.append(gs.score(K[np.ix_(te, tr)], y[te]))
    return float(np.mean(accs))


def hardware_nested(n_splits=20):
    """Re-score every IBM block with nested CV (exact kernel, unit diagonal,
    measured diagonal, per-point mitigated)."""
    out = {}
    for d in sorted(glob.glob(os.path.join(RES, "hardware", "*"))):
        name = os.path.basename(d)
        dev_task = name.split("_n")[0]
        task = dev_task.split("_", 2)[2]
        n = int(name.split("_n")[1].split("_")[0])
        for p in sorted(glob.glob(os.path.join(d, "gram_c*.npz"))):
            c = float(p.split("gram_c")[1][:-4])
            z = np.load(p)
            off, diag, y = z["off"], z["diag"], z["y"]
            N = len(diag)
            iu = np.triu_indices(N, 1)
            base = 2.0 ** -n
            K = np.eye(N)
            K[iu] = off
            K.T[iu] = off
            Km = K.copy()
            np.fill_diagonal(Km, diag)
            lam_i = np.sqrt(np.clip((diag - base) / (1 - base), 1e-6, None))
            ll = np.outer(lam_i, lam_i)
            Kmit = (K - (1 - ll) * base) / ll
            np.fill_diagonal(Kmit, 1.0)
            out[f"{name}|{c}"] = {
                "device": "_".join(name.split("_")[:2]), "task": task, "c": c,
                "nested_exact": _nested_cv(_exact_gram(task, n, c), y, n_splits),
                "nested_unit": _nested_cv(K, y, n_splits),
                "nested_meas": _nested_cv(Km, y, n_splits),
                "nested_mit": _nested_cv(Kmit, y, n_splits)}
    json.dump(out, open(os.path.join(RES, "hardware_nested.json"), "w"),
              indent=1)
    return out


def nested_stats():
    """Summary of the nested-CV hardware scores (digits task only; the
    quantum-native task is not learnable from 32 points)."""
    import pandas as pd
    r = json.load(open(os.path.join(RES, "hardware_nested.json")))
    df = pd.DataFrame(r.values())
    d = df[df.task == "digits3v5"].copy()
    d["gain"] = d.nested_meas - d.nested_unit
    d["mit_vs_meas"] = d.nested_mit - d.nested_meas
    out = {"digits_blocks": int(len(d)),
           "meas_better": int((d.gain > 0).sum()),
           "meas_worse": int((d.gain < 0).sum()),
           "gain_mean": float(d.gain.mean()),
           "gain_min": float(d.gain.min()), "gain_max": float(d.gain.max()),
           "mit_vs_meas_mean": float(d.mit_vs_meas.mean()),
           "mit_vs_meas_min": float(d.mit_vs_meas.min()),
           "mit_vs_meas_max": float(d.mit_vs_meas.max()),
           "best_c": {}, "rows": d.to_dict(orient="records")}
    for dev, g in d.groupby("device"):
        out["best_c"][dev] = {k: float(g.loc[g[k].idxmax(), "c"]) for k in
                              ("nested_exact", "nested_unit", "nested_meas")}
    json.dump(out, open(os.path.join(RES, "hardware_nested_stats.json"), "w"),
              indent=1)
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-nulls", action="store_true",
                        help="update hardware nulls from saved raw blocks")
    args = parser.parse_args()
    if args.refresh_nulls:
        summary = refresh_hardware_nulls()
        print(json.dumps(summary["hardware_stats"], indent=1))
    else:
        main()
