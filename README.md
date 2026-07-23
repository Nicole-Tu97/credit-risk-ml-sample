# Credit default prediction: a governed PD model

A probability-of-default model on the public UCI credit-card dataset, built end to end. The point of the project isn't the classifier on its own. It's everything a credit-risk team needs around it before a model can be trusted with a decision: leakage-safe features, honest validation, calibration, fairness checks, explainability, and a written account of where the model fails.

> Data: UCI "Default of Credit Card Clients" (Taiwan, 2005; 30,000 accounts). Public dataset, nothing proprietary.

## Data
30,000 accounts with 23 raw fields: credit limit, a few demographics, and six months of repayment status, bill amounts, and payments. The target is whether the account defaults the following month (22.1% base rate). A handful of undocumented `EDUCATION` and `MARRIAGE` category codes are folded into "other" and logged, rather than left to create phantom categories.

## Approach
**Features.** I turn the 23 raw fields into 36, adding utilization ratios, delinquency counts and severity, payment-to-bill coverage, and balance and delinquency trends. Every feature is a row-wise transform of pre-decision history, so there are no cross-row statistics, no target leakage, and no future information. The same transforms are also written in SQL (`sql/feature_extraction.sql`) so the features can be built directly in a warehouse.

**Models.** A logistic-regression model as an interpretable baseline, and a HistGradientBoosting model as the challenger. The challenger's probabilities are isotonic-calibrated, so a predicted PD can be read as an actual default rate.

**Validation.** I split off a 30% hold-out once and never touch it for fitting, tuning, or calibration. Hyperparameters come from 5-fold cross-validation on the training set, and I report both the CV AUC and the train-vs-test gap so overfitting is something you can see rather than take on trust.

## Results (30% hold-out)

| Model | AUC | Gini | KS | Brier | PSI |
|---|--:|--:|--:|--:|--:|
| Logistic regression (baseline) | 0.754 | 0.509 | 0.397 | 0.192 | 0.001 |
| **HistGradientBoosting + isotonic** | **0.784** | **0.569** | **0.429** | **0.134** | **0.001** |

5-fold cross-validated AUC on the training set is 0.787, effectively equal to the 0.784 hold-out. That CV-to-test gap of about 0.003 is the real evidence the model generalizes; the raw train-vs-test gap (0.031) looks larger only because it compares training-set resubstitution to test. Worth saying: on a dataset this well studied, an AUC much above 0.80 usually means something has leaked, so a credible 0.78 was the goal, not a bigger headline number.

(The PSI column above is a train-vs-test score-distribution check on a single random split, near zero by construction — not an out-of-time drift metric. Real drift monitoring is in the roadmap.)

![Champion model hold-out performance: ROC, precision-recall, calibration, score separation](outputs/model_performance.png)

## Explainability

Global drivers come from the logistic coefficients, permutation importance, and SHAP; the strongest by a wide margin is the most recent repayment status (`PAY_1`). Per-applicant SHAP values feed the adverse-action layer below.

![SHAP summary](outputs/shap_summary.png)

## Where it breaks (`outputs/findings.md`)

I spent as much time on where the model fails as on its headline number:
- Calibration is weakest in the high-risk tail, which is exactly where declines happen, so those cases should go to a person rather than an automatic decline.
- Discrimination is not uniform across segments, so one global cutoff is not equally fair or accurate for everyone.
- About 175 accounts scored below 10% PD still default (6% of that group). These confident false negatives are the quiet, expensive errors that pure auto-approval would wave through.

## Fairness & governance

Discrimination and decline rates are broken out by sex, age band, and education (`outputs/fairness.csv`). Decline-rate parity is fine on sex (0.86) but dips below the 4/5ths rule on age (0.79), and looks very low on education (0.14) — though that last figure is mostly an artifact of a tiny n=120 "other" group with almost no declines. I flag these as fairness risks rather than smooth them over. [`MODEL_CARD.md`](MODEL_CARD.md) records intended use, limitations, monitoring (PSI and calibration drift), human checkpoints, and how the work lines up with **OSFI E-23** and **FCAC**. The LLM adverse-action layer is deliberately constrained: it can cite only the model's actual top risk drivers for an applicant, so it cannot invent a reason the model did not use.

## Roadmap / next steps

These are the steps that would take this work sample from a point-in-time demo toward a model an origination team could actually deploy and defend.

**Validation & monitoring**
- Run a true out-of-time / vintage split instead of the random stratified hold-out: train on earlier origination months, test on later ones to expose temporal drift. Pair it with a feature-staleness ablation on `PAY_1` and `pay_max` (the two freshest, most dominant drivers) to bound refresh cadence. This is the single largest gap and should come first.
- Replace the monitoring prose with an executable drift monitor: persist a versioned training reference (score bins, per-feature bins, calibration curve) and score each new batch for feature PSI, score PSI, and a calibration-drift test (Spiegelhalter Z / ECE), emitting warn/fail at the documented thresholds. Current PSI is train-vs-test from a single split and is near zero by construction.
- Gate champion-challenger promotion in CI (minimum AUC lift, maximum overfit gap, maximum ECE, minimum fairness parity), and serialize the fitted pipeline, data checksum, pinned package versions, and seed to a run manifest, so any model swap is auditable and reproducible.

**Modeling**
- Fix the high-risk tail: benchmark isotonic against a monotone beta / Platt recalibration on a dedicated calibration split, bootstrap the calibration map for per-decile confidence bands, and map PD to a rating master scale. Isotonic is high-variance where the top decile thins out (0.718 predicted vs 0.700 observed).
- Add monotonic constraints (HistGBM `monotonic_cst`) forcing PD to rise with the delinquency and utilization features (`PAY_1`, `pay_max`, `n_delinq`, `util_*`). This steadies tail behavior and keeps the SHAP-based adverse-action reasons internally consistent, at near-zero AUC cost.
- Replace the blanket `fillna(0.0)` with missingness indicators and the GBM's native NaN handling. Filling with zero conflates "no history" with "current and zero utilization," biasing PD downward and likely feeding the confident false negatives.

**Fairness & governance**
- Report calibration per protected group (slope, intercept, ECE), not just per-segment AUC, and replace the single min/max decline-parity ratio with a reference-group adverse-impact ratio plus equalized-odds / equal-opportunity metrics, each with bootstrap CIs and a 4/5ths gate. This shows whether the weak education parity is real signal or small-sample noise (n=120).
- Ground the adverse-action reason codes in the deployed calibrated champion (they currently explain a separately refit uncalibrated GBM), and add a faithfulness test that the printed top-k reasons match the top-k SHAP drivers and that identical inputs yield identical notices.
- Add pre-registered fairness pass/fail gates, a subgroup fairness-drift monitor, and a named-owner sign-off to the governance pack. Route thin or low-AUC segments to human review using a PD uncertainty band (conformal / Venn-Abers interval width), not only a point-PD cutoff.

**Data & policy**
- Add reject inference before any origination use: the model is fit only on booked accounts, so it is biased relative to the through-the-door population it would actually score.
- Tie the cutoff to economics: optimize approve/decline bands on expected loss (PD × LGD × EAD, with `LIMIT_BAL` as an EAD proxy) and a good/default cost matrix, with segment-level bad-rate targets and a swap-set analysis, instead of the fixed 70th-percentile cutoff.
- Add a thin-file data-sufficiency flag that routes sparse applicants to a conservative policy, and reserve a documented slot for bureau-style attributes (tradeline count, inquiries, oldest-account age). The model rests on six months of `PAY_*`/`BILL_*` history, so a dormant applicant is scored low-risk by default.

## Run
```bash
pip install -r requirements.txt
python3 src/download_data.py      # fetch the public UCI dataset into data/
python3 src/pipeline.py           # build, validate, fairness, explainability, failure modes
python3 src/ai_reason_codes.py    # adverse-action notices (set ANTHROPIC_API_KEY to use the LLM)
```

## Structure
```
src/pipeline.py         model build, validation, fairness, explainability, failure-mode analysis
src/features.py         shared leakage-safe feature engineering
sql/feature_extraction.sql  the same feature logic in SQL, against a warehouse schema
src/ai_reason_codes.py  LLM adverse-action layer, grounded in the model's real SHAP drivers
outputs/                metrics.json, fairness.csv, findings.md, plots
MODEL_CARD.md           intended use, performance, limitations, monitoring, governance
```
