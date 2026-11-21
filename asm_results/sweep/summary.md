| Env | Predictor | Runs | Occ. frac (query) | Brier ↓ | BSS ↑ | Precision | Recall | F1 | Accuracy | Bal. acc | AUROC | Runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| env_A | constant_prior | 5 | 0.116 | 0.109 ± 0.019 | -0.07 ± 0.02 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.884 ± 0.021 | 0.50 ± 0.00 | 0.500 ± 0.000 | 0.0 ± 0.0 |
| env_A | nearest_observed | 5 | 0.116 | 0.044 ± 0.018 | 0.57 ± 0.13 | 0.85 ± 0.08 | 0.77 ± 0.08 | 0.80 ± 0.05 | 0.956 ± 0.018 | 0.87 ± 0.04 | 0.872 ± 0.039 | 0.3 ± 0.3 |
| env_A | gp | 5 | 0.116 | 0.019 ± 0.006 | 0.81 ± 0.05 | 0.90 ± 0.04 | 0.88 ± 0.08 | 0.88 ± 0.04 | 0.974 ± 0.007 | 0.93 ± 0.04 | 0.994 ± 0.005 | 23.4 ± 2.8 |
| env_A | tabpfn | 5 | 0.116 | 0.045 ± 0.012 | 0.55 ± 0.14 | 0.90 ± 0.04 | 0.53 ± 0.15 | 0.65 ± 0.12 | 0.939 ± 0.016 | 0.76 ± 0.07 | 0.963 ± 0.040 | 3.5 ± 0.8 |
| env_B | constant_prior | 5 | 0.110 | 0.102 ± 0.016 | -0.04 ± 0.02 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.890 ± 0.018 | 0.50 ± 0.00 | 0.500 ± 0.000 | 0.0 ± 0.0 |
| env_B | nearest_observed | 5 | 0.110 | 0.082 ± 0.026 | 0.16 ± 0.27 | 0.65 ± 0.16 | 0.63 ± 0.08 | 0.63 ± 0.09 | 0.918 ± 0.026 | 0.79 ± 0.04 | 0.794 ± 0.042 | 0.4 ± 0.4 |
| env_B | gp | 5 | 0.110 | 0.037 ± 0.009 | 0.62 ± 0.11 | 0.90 ± 0.03 | 0.67 ± 0.11 | 0.77 ± 0.08 | 0.956 ± 0.010 | 0.83 ± 0.05 | 0.940 ± 0.048 | 28.0 ± 9.8 |
| env_B | tabpfn | 5 | 0.110 | 0.055 ± 0.013 | 0.42 ± 0.18 | 0.86 ± 0.14 | 0.44 ± 0.11 | 0.57 ± 0.12 | 0.931 ± 0.012 | 0.71 ± 0.06 | 0.887 ± 0.086 | 3.0 ± 0.2 |
| all | constant_prior | 10 | 0.113 | 0.106 ± 0.017 | -0.06 ± 0.02 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.887 ± 0.019 | 0.50 ± 0.00 | 0.500 ± 0.000 | 0.0 ± 0.0 |
| all | nearest_observed | 10 | 0.113 | 0.063 ± 0.029 | 0.36 ± 0.30 | 0.75 ± 0.16 | 0.70 ± 0.10 | 0.72 ± 0.11 | 0.937 ± 0.029 | 0.83 ± 0.06 | 0.833 ± 0.056 | 0.3 ± 0.4 |
| all | gp | 10 | 0.113 | 0.028 ± 0.012 | 0.72 ± 0.13 | 0.90 ± 0.03 | 0.77 ± 0.14 | 0.82 ± 0.09 | 0.965 ± 0.013 | 0.88 ± 0.07 | 0.967 ± 0.043 | 25.7 ± 7.2 |
| all | tabpfn | 10 | 0.113 | 0.050 ± 0.013 | 0.49 ± 0.17 | 0.88 ± 0.10 | 0.48 ± 0.13 | 0.61 ± 0.12 | 0.935 ± 0.014 | 0.74 ± 0.07 | 0.925 ± 0.075 | 3.3 ± 0.6 |

Paired Brier comparison on identical runs (wins / runs, mean Δ Brier):

- tabpfn vs gp: tabpfn lower Brier in 0/10 runs, mean Δ +0.0220
- tabpfn vs nearest_observed: tabpfn lower Brier in 7/10 runs, mean Δ -0.0132
- gp vs nearest_observed: gp lower Brier in 10/10 runs, mean Δ -0.0352

## Interpretation (env_A + env_B, seeds 0-4, 10 runs)

- **Leakage:** none found. Training labels come only from the LiDAR grid, query cells are
  exactly the unobserved cells (asserted disjoint per run), and ground truth is used only for
  scoring. All predictors use the same 2000 training cells per run.
- **Class imbalance:** 11.3 % of the unobserved cells are occupied, so "always free"
  (constant_prior) already reaches 0.887 accuracy. Accuracy is therefore not informative;
  Brier, BSS, F1, balanced accuracy and AUROC are. The training cells are only 3.9 % occupied
  (box surfaces seen by LiDAR, box interiors unseen), a prior shift that pushes probabilistic
  classifiers toward "free" at the 0.5 threshold.
- **TabPFN vs GP:** GP has lower Brier in 10/10 paired runs (mean Δ 0.022) and higher AUROC
  (0.967 vs 0.925), so the gap is not just a threshold/calibration effect. TabPFN keeps high
  precision but low recall (0.48) and shows axis-aligned streak artefacts on raw (x, y) inputs.
- **TabPFN vs NearestObserved:** TabPFN lower Brier in 7/10 runs; a modest gain, mostly in env_B.
- **Runtime:** TabPFN ~3 s per run (cloud API, includes network), GP ~26 s (sklearn, 2000
  points, hyperparameter optimisation), NearestObserved < 1 s.
- **Conclusion:** on this task TabPFN gives no meaningful advantage over a GP. Its only edge
  is runtime, which comes from the cloud. Single fixed TabPFN config, raw coordinates as
  features, no training rebalancing; those are the obvious knobs before ruling it out.
