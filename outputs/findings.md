# Findings — performance, validity, and where the model breaks

## Validity controls (no leakage, no hidden overfitting)
- **Feature engineering is leakage-safe:** all 13 engineered features are per-row transforms of
  historical, pre-decision columns (repayment status, bills, payments). No cross-row statistics, no target, no future data.
- **Protected attributes excluded from the model:** SEX, AGE, EDUCATION, MARRIAGE are dropped from the 32-feature
  matrix (disparate-treatment control) and retained only as axes for the fairness audit below — the model never sees them.
- **The test set was held out once** and used only for the final evaluation below — never for fitting, feature stats,
  tuning, or calibration.
- **Tuning on train only:** hyperparameters chosen by 5-fold CV on the training set (CV AUC = 0.7869).
- **Calibration on train only:** isotonic calibration fit inside the training set via CV, then measured on test.
- **Overfitting is measured, not assumed:** the honest check is CV vs. test — 5-fold CV AUC (0.7869) vs. test AUC
  (0.7814), a gap of ~0.006. (The raw train-vs-test gap of 0.0311 looks larger
  only because it compares training-set resubstitution to test.)

## Performance (30% hold-out)
| model | AUC | Gini | KS | Brier | PSI | train AUC | gap |
|---|--:|--:|--:|--:|--:|--:|--:|
| LogisticRegression (baseline) | 0.7495 | 0.4991 | 0.3961 | 0.1427 | 0.0007 | 0.7609 | 0.0114 |
| HistGradientBoosting + isotonic (champion) | 0.7814 | 0.5627 | 0.4278 | 0.1348 | 0.0024 | 0.8124 | 0.0311 |

Champion: **HistGradientBoosting + isotonic**. It leads the baseline on discrimination (AUC/Gini/KS) at a comparable, well-calibrated Brier —
both models are calibrated, so the Brier comparison is apples-to-apples — while keeping the CV-to-test gap tiny.
(Note: on this well-known dataset an AUC far above ~0.80 would signal leakage —
the aim here is a *credible* lift, not a vanity number. PSI here is a train-vs-test score-distribution check on a
single random split, ≈ 0 by construction — not an out-of-time drift metric; see the roadmap in the README.)

## Where it still breaks
1. **High-risk-tail calibration** — observed-minus-predicted default by PD decile:
        pred_PD  actual    gap     n
decile                              
0         0.033   0.042  0.009   900
1         0.062   0.068  0.006   900
2         0.086   0.081 -0.005   900
3         0.114   0.121  0.007   900
4         0.142   0.147  0.005   903
5         0.168   0.185  0.016  1051
6         0.197   0.171 -0.025   747
7         0.269   0.275  0.006   899
8         0.422   0.424  0.002   900
9         0.717   0.696 -0.022   900
   Calibration is much improved but the extreme tail remains hardest — route those to human review, not auto-decline.
2. **Uneven segment discrimination** — weakest: EDUCATION=other (AUC=0.616, n=120)
   vs overall 0.7814. Decline-rate parity ratios: {'AGE_band': 0.801, 'EDUCATION': 0.423, 'MARRIAGE': 0.91, 'SEX': 0.882}.
3. **Confidently-approved defaulters** — 174 accounts scored PD<10% still defaulted (6% of that group).

## Top drivers (GBM permutation importance)
Computed on the uncalibrated base of the isotonic champion; isotonic calibration is a monotone transform of the
score, so per-applicant driver rankings are unchanged.

pay_max        0.0501
PAY_1          0.0232
util_max       0.0054
bill_trend     0.0051
util_recent    0.0047
pay_amt_sum    0.0037
pay_to_bill    0.0023
bill_mean      0.0022
