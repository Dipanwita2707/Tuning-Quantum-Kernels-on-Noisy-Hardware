"""Verify manuscript claim inputs against recorded scores, without refitting.

Only stored CSV/JSON summaries are read. No SVM, quantum simulation, hardware
service, permutation test, or bootstrap is run. Only a verification summary is printed.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
DATA = ROOT / 'data/extended'
checks = []


def verify(name, observed, expected, atol=1e-11):
    if isinstance(observed, dict):
        ok = set(observed) == set(expected) and all(
            np.isclose(observed[k], expected[k], atol=atol, rtol=0)
            for k in observed
        )
    else:
        ok = bool(np.allclose(observed, expected, atol=atol, rtol=0))
    checks.append({'claim': name, 'matches_recorded_evidence': bool(ok)})
    if not ok:
        raise ValueError(f'{name}: recomputed {observed!r}, recorded {expected!r}')


def load(name):
    return json.loads((DATA / name).read_text())


scores = pd.read_csv(DATA / 'sim_scores.csv')
selected = pd.read_csv(DATA / 'sim_selection.csv')
tests = load('sim_tests.json')
extra = load('sim_exploratory.json')
verify('Configuration and bandwidth counts', [len(scores), len(selected)], [5760, 768])
verify('Dataset/seed cluster count', selected.groupby(['dataset', 'seed']).ngroups, 64)
verify('All H3 clusters contain 3 maps x 4 noise settings',
       selected.groupby(['dataset', 'seed']).size().to_numpy(), np.full(64, 12))

lookup = selected.set_index(['fmap', 'dataset', 'seed', 'noise'])
alignment_measured = []
rebuilt_rows = []
for key, group in scores.groupby(['fmap', 'dataset', 'seed']):
    ideal = group[group.noise == 'ideal']
    ideal_ties = ideal[np.isclose(ideal.cv_unit, ideal.cv_unit.max(), rtol=0, atol=1e-12)]
    ideal_log = np.log2(ideal_ties.c).mean()
    transferred = set(ideal[np.isclose(ideal.cv_unit, ideal.cv_unit.max(), atol=1e-12)].c)
    for noise, noisy in group[group.noise != 'ideal'].groupby('noise'):
        actual = lookup.loc[key + (noise,)]
        values = {}
        for suffix in ['unit', 'meas']:
            ties = noisy[np.isclose(noisy['cv_' + suffix],
                                    noisy['cv_' + suffix].max(), rtol=0, atol=1e-12)]
            values['shift_' + suffix] = float(np.log2(ties.c).mean() - ideal_log)
            values['acc_dev_cv_' + suffix] = float(ties['acc_' + suffix].mean())
        kta_ties = noisy[np.isclose(noisy.kta_unit, noisy.kta_unit.max(), rtol=0, atol=1e-12)]
        values.update({
            'acc_sim_cv': float(noisy[noisy.c.isin(transferred)].acc_unit.mean()),
            'acc_kta': float(kta_ties.acc_unit.mean()),
            'shift_kta': float(np.log2(kta_ties.c).mean() - ideal_log),
            'acc_oracle': float(noisy.acc_unit.max()),
            'asym': float(noisy.asym.mean()),
            'excess_loss': float((noisy.acc_affine - noisy.acc_unit).mean()),
            'corr_exact': float(noisy.corr_exact.mean()),
        })
        verify('Selection reconstruction ' + '|'.join(map(str, key + (noise,))),
               values, {k: float(actual[k]) for k in values})
        rebuilt_rows.append(dict(zip(['fmap', 'dataset', 'seed'], key),
                                 noise=noise, **values))
        measured_ties = noisy[np.isclose(noisy.kta_meas, noisy.kta_meas.max(),
                                         rtol=0, atol=1e-12)]
        alignment_measured.append(float(measured_ties.acc_meas.mean()))

# Reproduce the published pipeline's in-memory aggregates. Reading its derived
# selection CSV back can perturb rank ties at machine precision.
selected = pd.DataFrame(rebuilt_rows)
cluster = selected.groupby(['dataset', 'seed'])
shift = cluster.shift_unit.mean()
verify('H1 mean and up/down/tied counts',
       [shift.mean(), (shift > 0).sum(), (shift < 0).sum(), (shift == 0).sum()],
       [tests['H1_drift']['mean_log2_shift'], 50, 11, 3])
verify('H1 noise-specific means', selected.groupby('noise').shift_unit.mean().to_dict(),
       tests['H1_drift']['by_noise'])
verify('H1 feature-map means', selected.groupby('fmap').shift_unit.mean().to_dict(),
       extra['drift_by_fmap'])
verify('H1 dataset means', selected.groupby('dataset').shift_unit.mean().to_dict(),
       extra['dataset_level']['h1_shift_by_dataset'])
verify('H2 mean absolute shifts',
       [selected.shift_unit.abs().mean(), selected.shift_meas.abs().mean()],
       [tests['H2_diag_reduces_drift']['mean_abs_shift_unit'],
        tests['H2_diag_reduces_drift']['mean_abs_shift_meas']])
verify('H3 pooled correlation of bandwidth-averaged quantities',
       spearmanr(selected.asym, selected.excess_loss).statistic,
       tests['H3_asym_vs_excess_loss']['spearman'])
for noise, group in selected.groupby('noise'):
    verify('H3 means by noise ' + noise,
           [group.asym.mean(), group.excess_loss.mean()],
           [tests['H3_asym_vs_excess_loss']['by_noise'][noise]['asym'],
            tests['H3_asym_vs_excess_loss']['by_noise'][noise]['excess_loss']])
delta = selected.acc_dev_cv_meas - selected.acc_sim_cv
verify('H4 mean contrast', delta.groupby([selected.dataset, selected.seed]).mean().mean(),
       tests['H4_protocol']['mean_diff'])
verify('H4 dataset contrasts', delta.groupby(selected.dataset).mean().to_dict(),
       extra['dataset_level']['h4_diff_by_dataset'])
for name, expected in tests['rule_means'].items():
    verify('Rule mean ' + name, selected[name].mean(), expected)
verify('Secondary measured-alignment complete procedure', np.mean(alignment_measured),
       load('sim_kta_meas.json')['acc_kta_meas'])

summary = load('summary_ext.json')
hw = pd.DataFrame(summary['hardware_runs'].values())
nested = pd.DataFrame(load('hardware_nested.json').values())
digits = nested[nested.task == 'digits3v5']
gain = digits.nested_meas - digits.nested_unit
ns = load('hardware_nested_stats.json')
verify('Digits diagonal benefit',
       [(gain > 0).sum(), gain.mean(), gain.min(), gain.max()],
       [ns['meas_better'], ns['gain_mean'], ns['gain_min'], ns['gain_max']])
mit = digits.nested_mit - digits.nested_meas
verify('Digits mitigation relative to measured diagonal',
       [mit.mean(), mit.min(), mit.max()],
       [ns['mit_vs_meas_mean'], ns['mit_vs_meas_min'], ns['mit_vs_meas_max']])
verify('IBM symmetry exceedances and affine-parameter comparisons',
       [(hw.asym_index > hw.asym_null_975).sum(),
        (hw.asym_index > 1.36).sum(),
        (hw.lambda_diag_affine < hw.lambda_offdiag_slope).sum(),
        (hw.lambda_diag_affine < hw.lambda_offdiag).sum()], [16, 14, 13, 16])
merged = hw.merge(nested, on=['device', 'task', 'c'])
rho = spearmanr(merged.diag_heterogeneity, merged.nested_meas - merged.nested_unit)
verify('Corrected nested-C diagonal correlation', [rho.statistic, rho.pvalue],
       [0.06917915222369332, 0.785039384345873])

report = {'checks_passed': len(checks), 'checks_failed': 0,
          'scope': 'Recorded score reconstruction and summary consistency; no new experiments or fits',
          'h4_comparison': 'Device-selected measured diagonal versus simulator-selected unit diagonal',
          'alignment_comparison': 'Both selection criterion and final SVM diagonal change',
          'rank_precision': 'Primary correlations use in-memory reconstruction from per-bandwidth scores; reloading selection CSV can perturb ties',
          'primary_p_values_and_intervals': 'Preserved recorded estimates; this lightweight verification does not regenerate p-values or intervals',
          'checks': checks}
print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, indent=2))
