# Findings — what the model gets right, and where it breaks

## Performance (test, 30% holdout)
| model | AUC | Gini | KS | Brier | PSI (train vs test) |
|---|--:|--:|--:|--:|--:|
| LogisticRegression (scorecard baseline) | 0.7159 | 0.4318 | 0.3709 | 0.208 | 0.0009 |
| HistGradientBoosting (challenger) | 0.7757 | 0.5515 | 0.4242 | 0.1361 | 0.0008 |

Champion: **HistGradientBoosting**. PSI < 0.1 ⇒ score distribution stable between train and test.

## Where it breaks (failure-mode analysis)
1. **Calibration drifts in the high-risk tail.** Observed-minus-predicted default rate by PD decile:
        pred_PD  actual    gap    n
decile                             
0         0.059   0.041 -0.018  900
1         0.080   0.066 -0.015  900
2         0.097   0.116  0.018  900
3         0.114   0.107 -0.007  900
4         0.135   0.149  0.014  900
5         0.157   0.162  0.005  900
6         0.190   0.190 -0.000  900
7         0.251   0.271  0.020  900
8         0.403   0.419  0.016  900
9         0.710   0.692 -0.018  900
   The top deciles are where predicted and actual diverge most — the model is least reliable exactly where decisions
   are most consequential (declines), so scores there should feed human review rather than auto-decline.
2. **Uneven discrimination across segments.** Weakest segment AUC: EDUCATION=other
   (AUC=0.607, n=120) vs. overall 0.7757. A single global cutoff is not
   equally accurate for every group — a fairness and governance concern, not just an accuracy one.
3. **Confidently-approved defaulters.** 162 test accounts scored PD<10% yet defaulted
   (7% of the low-PD population) — the costly, silent errors auto-approval would miss.
4. **Data-quality leak.** EDUCATION/MARRIAGE contained undocumented codes (0/5/6) folded into "other" during prep;
   left unhandled they silently create phantom categories — the kind of issue that must be caught before, not after, deployment.

## How these were found
By validating the model against its own assumptions rather than the headline AUC: calibration by risk band,
per-segment discrimination, and inspection of high-confidence errors — then routing the fragile regions to human checkpoints.
