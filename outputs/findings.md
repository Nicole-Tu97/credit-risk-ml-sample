# Findings — performance, validity, and where the model breaks

## Validity controls (no leakage, no hidden overfitting)
- **Feature engineering is leakage-safe:** all 13 engineered features are per-row transforms of
  historical, pre-decision columns (repayment status, bills, payments). No cross-row statistics, no target, no future data.
- **The test set was held out once** and used only for the final evaluation below — never for fitting, feature stats,
  tuning, or calibration.
- **Tuning on train only:** hyperparameters chosen by 5-fold CV on the training set (CV AUC = 0.7874).
- **Calibration on train only:** isotonic calibration fit inside the training set via CV, then measured on test.
- **Overfitting is measured, not assumed:** the honest check is CV vs. test — 5-fold CV AUC (0.7874) vs. test AUC
  (0.7844), a gap of ~0.003. (The raw train-vs-test gap of 0.0308 looks larger
  only because it compares training-set resubstitution to test.)

## Performance (30% hold-out)
| model | AUC | Gini | KS | Brier | PSI | train AUC | gap |
|---|--:|--:|--:|--:|--:|--:|--:|
| LogisticRegression (baseline) | 0.7544 | 0.5089 | 0.3968 | 0.1922 | 0.0007 | 0.7649 | 0.0104 |
| HistGradientBoosting + isotonic (champion) | 0.7844 | 0.5687 | 0.4286 | 0.1343 | 0.0005 | 0.8151 | 0.0308 |

Champion: **HistGradientBoosting + isotonic**. Feature engineering, tuning, and calibration improved discrimination and Brier while
keeping the CV-to-test gap tiny. (Note: on this well-known dataset an AUC far above ~0.80 would signal leakage —
the aim here is a *credible* lift, not a vanity number. PSI here is a train-vs-test score-distribution check on a
single random split, ≈ 0 by construction — not an out-of-time drift metric; see the roadmap in the README.)

## Where it still breaks
1. **High-risk-tail calibration** — observed-minus-predicted default by PD decile:
        pred_PD  actual    gap    n
decile                             
0         0.033   0.040  0.007  902
1         0.061   0.069  0.008  898
2         0.086   0.084 -0.002  900
3         0.113   0.111 -0.001  906
4         0.143   0.133 -0.010  895
5         0.167   0.188  0.021  899
6         0.192   0.189 -0.004  901
7         0.267   0.273  0.006  900
8         0.424   0.425  0.002  901
9         0.718   0.700 -0.017  898
   Calibration is much improved but the extreme tail remains hardest — route those to human review, not auto-decline.
2. **Uneven segment discrimination** — weakest: EDUCATION=other (AUC=0.612, n=120)
   vs overall 0.7844. Decline-rate parity ratios: {'AGE_band': np.float64(0.792), 'EDUCATION': np.float64(0.139), 'SEX': np.float64(0.86)}.
3. **Confidently-approved defaulters** — 175 accounts scored PD<10% still defaulted (6% of that group).

## Top drivers (GBM permutation importance)
pay_max        0.0489
PAY_1          0.0226
util_max       0.0055
bill_trend     0.0044
util_recent    0.0043
pay_amt_sum    0.0037
bill_mean      0.0021
pay_to_bill    0.0021
