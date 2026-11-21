| Env | Predictor | Runs | Occ. frac (query) | Brier ↓ | BSS ↑ | Precision | Recall | F1 | Accuracy | Bal. acc | AUROC | Runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| env_A | nearest_observed | 5 | 0.116 | 0.044 ± 0.018 | 0.57 ± 0.13 | 0.85 ± 0.08 | 0.77 ± 0.08 | 0.80 ± 0.05 | 0.956 ± 0.018 | 0.87 ± 0.04 | 0.872 ± 0.039 | 0.2 ± 0.0 |
| env_A | gp | 5 | 0.116 | 0.019 ± 0.006 | 0.81 ± 0.05 | 0.90 ± 0.04 | 0.88 ± 0.08 | 0.88 ± 0.04 | 0.974 ± 0.007 | 0.93 ± 0.04 | 0.994 ± 0.005 | 22.3 ± 1.2 |
| env_A | gp_bal | 5 | 0.116 | 0.069 ± 0.013 | 0.32 ± 0.08 | 0.66 ± 0.07 | 0.98 ± 0.03 | 0.78 ± 0.05 | 0.935 ± 0.029 | 0.95 ± 0.02 | 0.994 ± 0.003 | 0.2 ± 0.0 |
| env_A | tabpfn | 5 | 0.116 | 0.045 ± 0.012 | 0.55 ± 0.14 | 0.90 ± 0.04 | 0.53 ± 0.15 | 0.65 ± 0.12 | 0.939 ± 0.016 | 0.76 ± 0.07 | 0.963 ± 0.040 | 3.1 ± 0.7 |
| env_A | tabpfn_bal | 5 | 0.116 | 0.049 ± 0.017 | 0.52 ± 0.16 | 0.76 ± 0.10 | 0.73 ± 0.28 | 0.71 ± 0.15 | 0.937 ± 0.024 | 0.85 ± 0.13 | 0.965 ± 0.028 | 3.1 ± 0.6 |
| env_A | tabpfn_feat | 5 | 0.116 | 0.038 ± 0.010 | 0.62 ± 0.09 | 0.90 ± 0.04 | 0.62 ± 0.17 | 0.72 ± 0.13 | 0.947 ± 0.021 | 0.80 ± 0.08 | 0.984 ± 0.006 | 3.1 ± 0.6 |
| env_A | tabpfn_feat_bal | 5 | 0.116 | 0.052 ± 0.009 | 0.48 ± 0.06 | 0.75 ± 0.09 | 0.65 ± 0.21 | 0.68 ± 0.09 | 0.932 ± 0.013 | 0.81 ± 0.09 | 0.941 ± 0.031 | 3.0 ± 0.6 |
| env_B | nearest_observed | 5 | 0.110 | 0.082 ± 0.026 | 0.16 ± 0.27 | 0.65 ± 0.16 | 0.63 ± 0.08 | 0.63 ± 0.09 | 0.918 ± 0.026 | 0.79 ± 0.04 | 0.794 ± 0.042 | 0.2 ± 0.1 |
| env_B | gp | 5 | 0.110 | 0.037 ± 0.009 | 0.62 ± 0.11 | 0.90 ± 0.03 | 0.67 ± 0.11 | 0.77 ± 0.08 | 0.956 ± 0.010 | 0.83 ± 0.05 | 0.940 ± 0.048 | 23.7 ± 3.8 |
| env_B | gp_bal | 5 | 0.110 | 0.078 ± 0.010 | 0.19 ± 0.14 | 0.67 ± 0.07 | 0.78 ± 0.15 | 0.72 ± 0.09 | 0.935 ± 0.016 | 0.87 ± 0.07 | 0.899 ± 0.086 | 0.6 ± 0.3 |
| env_B | tabpfn | 5 | 0.110 | 0.055 ± 0.013 | 0.42 ± 0.18 | 0.86 ± 0.14 | 0.44 ± 0.11 | 0.57 ± 0.12 | 0.931 ± 0.012 | 0.71 ± 0.06 | 0.887 ± 0.086 | 3.1 ± 0.7 |
| env_B | tabpfn_bal | 5 | 0.110 | 0.069 ± 0.035 | 0.27 ± 0.43 | 0.67 ± 0.18 | 0.59 ± 0.14 | 0.62 ± 0.14 | 0.919 ± 0.034 | 0.77 ± 0.07 | 0.864 ± 0.107 | 3.0 ± 0.1 |
| env_B | tabpfn_feat | 5 | 0.110 | 0.065 ± 0.043 | 0.32 ± 0.50 | 0.80 ± 0.22 | 0.55 ± 0.05 | 0.64 ± 0.11 | 0.930 ± 0.030 | 0.77 ± 0.03 | 0.898 ± 0.076 | 3.3 ± 0.6 |
| env_B | tabpfn_feat_bal | 5 | 0.110 | 0.066 ± 0.029 | 0.31 ± 0.35 | 0.70 ± 0.16 | 0.49 ± 0.14 | 0.58 ± 0.15 | 0.922 ± 0.022 | 0.73 ± 0.07 | 0.846 ± 0.113 | 5.5 ± 5.4 |
| all | nearest_observed | 10 | 0.113 | 0.063 ± 0.029 | 0.36 ± 0.30 | 0.75 ± 0.16 | 0.70 ± 0.10 | 0.72 ± 0.11 | 0.937 ± 0.029 | 0.83 ± 0.06 | 0.833 ± 0.056 | 0.2 ± 0.1 |
| all | gp | 10 | 0.113 | 0.028 ± 0.012 | 0.72 ± 0.13 | 0.90 ± 0.03 | 0.77 ± 0.14 | 0.82 ± 0.09 | 0.965 ± 0.013 | 0.88 ± 0.07 | 0.967 ± 0.043 | 23.0 ± 2.8 |
| all | gp_bal | 10 | 0.113 | 0.074 ± 0.012 | 0.26 ± 0.13 | 0.67 ± 0.07 | 0.88 ± 0.14 | 0.75 ± 0.08 | 0.935 ± 0.022 | 0.91 ± 0.07 | 0.947 ± 0.076 | 0.4 ± 0.3 |
| all | tabpfn | 10 | 0.113 | 0.050 ± 0.013 | 0.49 ± 0.17 | 0.88 ± 0.10 | 0.48 ± 0.13 | 0.61 ± 0.12 | 0.935 ± 0.014 | 0.74 ± 0.07 | 0.925 ± 0.075 | 3.1 ± 0.7 |
| all | tabpfn_bal | 10 | 0.113 | 0.059 ± 0.028 | 0.40 ± 0.33 | 0.71 ± 0.15 | 0.66 ± 0.22 | 0.66 ± 0.14 | 0.928 ± 0.029 | 0.81 ± 0.10 | 0.914 ± 0.091 | 3.0 ± 0.4 |
| all | tabpfn_feat | 10 | 0.113 | 0.051 ± 0.033 | 0.47 ± 0.38 | 0.85 ± 0.16 | 0.58 ± 0.12 | 0.68 ± 0.12 | 0.939 ± 0.026 | 0.78 ± 0.06 | 0.941 ± 0.068 | 3.2 ± 0.6 |
| all | tabpfn_feat_bal | 10 | 0.113 | 0.059 ± 0.022 | 0.40 ± 0.25 | 0.73 ± 0.13 | 0.57 ± 0.18 | 0.63 ± 0.13 | 0.927 ± 0.018 | 0.77 ± 0.09 | 0.893 ± 0.093 | 4.3 ± 3.8 |

Paired Brier comparison on identical runs (wins / runs, mean Δ Brier):

- tabpfn vs gp: tabpfn lower Brier in 0/10 runs, mean Δ +0.0220
- tabpfn vs nearest_observed: tabpfn lower Brier in 7/10 runs, mean Δ -0.0132
- tabpfn_bal vs gp: tabpfn_bal lower Brier in 0/10 runs, mean Δ +0.0311
- tabpfn_bal vs nearest_observed: tabpfn_bal lower Brier in 6/10 runs, mean Δ -0.0041
- tabpfn_feat vs gp: tabpfn_feat lower Brier in 0/10 runs, mean Δ +0.0235
- tabpfn_feat vs nearest_observed: tabpfn_feat lower Brier in 7/10 runs, mean Δ -0.0117
- tabpfn_feat_bal vs gp: tabpfn_feat_bal lower Brier in 0/10 runs, mean Δ +0.0312
- tabpfn_feat_bal vs nearest_observed: tabpfn_feat_bal lower Brier in 4/10 runs, mean Δ -0.0040
- gp vs nearest_observed: gp lower Brier in 10/10 runs, mean Δ -0.0352

## Interpretation: can TabPFN be rescued? (env_A + env_B, seeds 0-4, 10 runs)

Variants (same LiDAR observations and the same 2000 shared training cells per run):
- `*_bal`: fit on all occupied cells + 3x as many random free cells (raw probabilities).
- `tabpfn_feat`: x, y plus distance to nearest observed occupied / free cell and their counts
  within 0.35 m, computed from training cells only (leave-one-out for training rows).
- `gp_feat` was tried on env_A seed 0 only: Brier 0.122, AUROC 0.743, 111 s per run; dropped
  from the sweep for runtime. Query cells (box interiors, unscanned corners) have feature
  values outside the training range, which hurts the ARD GP more than TabPFN.

Findings:
- No TabPFN variant beats the plain GP in a single run (0/10 for every variant, mean
  Brier gap +0.022 to +0.031). Best TabPFN AUROC is tabpfn_feat at 0.941 vs GP 0.967.
- Rebalancing trades precision for recall as expected (tabpfn recall 0.48 -> 0.66) but
  worsens Brier, since the probabilities are no longer calibrated to the 11 % query rate.
- Context features help TabPFN slightly (recall 0.48 -> 0.58, AUROC 0.925 -> 0.941, removes
  most axis-aligned streaks) but do not close the gap to the GP.
- gp_bal is a notable by-product: 0.4 s per run (about 250 training points instead of 2000),
  recall 0.88 and AUROC 0.947, at the cost of a worse Brier from the shifted prior.
- Conclusion: on this occupancy-completion task, TabPFN gives no meaningful advantage over a
  GP on raw coordinates, even with rebalancing or context features.
