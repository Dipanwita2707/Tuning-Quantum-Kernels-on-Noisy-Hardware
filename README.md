# Quantum-kernel methods and recorded results

Authors: Dipanwita Kundu and Sourav Mukhopadhyay.
Department of Computer Science and Engineering, Lovely Professional University,
Phagwara, Punjab.

This repository contains the implementation and saved evidence for bandwidth
transfer and forward/reversed execution diagnostics in quantum kernels.

## Contents

- `src/ext_engine.py`: datasets, feature maps, noise models, and simulation engine.
- `src/analyze_ext_sim.py`: SVM scoring, bandwidth selection, and H1–H4 analysis.
- `src/hardware_ext.py`: IBM acquisition implementation.
- `src/analyze_ext.py`: hardware diagnostics and accuracy analysis.
- `src/oq_diag.py`, `src/openquantum_hw.py`: exploratory IQM acquisition implementation.
- `src/figures_ext.py`: optional plots generated from saved summaries.
- `src/verify_recorded_results.py`: lightweight recorded-result consistency checks.
- `data/extended/q4/`: 960 recorded four-qubit configurations, six bandwidths each.
- `data/extended/hardware/`: 18 saved IBM Gram blocks and job identifiers.
- `data/extended/oq_diag/`: exploratory IQM counts, entries, and job identifiers.
- `data/extended/*.csv` and `*.json`: recorded scores, selections, and summaries.

The public package excludes manuscripts, PDFs, bibliography files, review
documents, private context, preliminary-study files, and execution logs.

## Check recorded results without new experiments

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scipy
python src/verify_recorded_results.py
```

The verification reads saved CSV/JSON files and checks 793 numerical
relationships. It performs no quantum simulation, hardware acquisition, SVM
refit, permutation test, or bootstrap. The saved raw blocks permit deeper
independent checks; this lightweight command alone does not validate every raw
count or establish that the evaluation is unbiased.

For optional full analysis, `requirements.txt` records the main Qiskit 1.4.6
environment. IQM acquisition code instead uses a separate Qiskit 2 environment;
it is not required to inspect or verify the recorded results. Hardware
credentials are not included. Acquisition scripts can submit paid jobs when
explicitly invoked with submission/run options; no acquisition is part of the
verification command above.

## Result provenance and interpretation

`sim_scores.csv` contains 5,760 configuration–bandwidth evaluations.
`sim_selection.csv`, `sim_tests.json`, and `sim_exploratory.json` contain
selection contrasts and statistical summaries. Reproduce exact rank correlations
through the original in-memory aggregation from per-bandwidth scores: rounding
on reloading the derived selection CSV can change rank ties slightly.

H4 compares device-selected measured-diagonal scores against simulator-selected
unit-diagonal scores; both bandwidth choice and final diagonal treatment change.
The secondary alignment comparison also changes both components. Neither
contrast isolates the effect of bandwidth selection alone.

`summary_ext.json` includes hardware diagnostics and older non-nested scores.
Use `hardware_nested.json` and `hardware_nested_stats.json` for the reported
nested-C hardware comparison. These remain descriptive transductive evaluations:
preprocessing is fixed, whole-matrix eigenvalue clipping precedes splitting, and
bandwidth selection is not nested. The code preserves that historical procedure.

The corrected IBM symmetry null pools forward and reversed probabilities,
uses 10,000 binomial redraws per block, and yields 16 of 18 pointwise threshold
exceedances. Fixed execution order confounds directional errors with temporal
drift; thresholds do not adjust for multiple blocks. Measurements come from one
day per IBM processor, and IQM results are exploratory. No quantum advantage,
causal error attribution, or general prediction of accuracy loss is established.

The archive contains existing measurements and analyses. Packaging and
verification do not create new experimental results.
