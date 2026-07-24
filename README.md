# Credit default prediction: a governed PD model

A probability-of-default model on the public UCI credit-card dataset, built end to end. The point of the project isn't the classifier on its own. It's everything a credit-risk team needs around it before a model can be trusted with a decision: leakage-safe features, honest validation, calibration, fairness checks, explainability, and a written account of where the model fails.

> Data: UCI "Default of Credit Card Clients" (Taiwan, 2005; 30,000 accounts). Public dataset, nothing proprietary.

## Data
30,000 accounts with 23 raw fields: credit limit, a few demographics, and six months of repayment status, bill amounts, and payments. The target is whether the account defaults the following month (22.1% base rate). A handful of undocumented `EDUCATION` and `MARRIAGE` category codes are folded into "other" and logged, rather than left to create phantom categories.

## Approach
**Features.** I turn six months of raw history into 13 engineered features — utilization ratios, delinquency counts and severity, payment-to-bill coverage, and balance and delinquency trends. The model trains on 32 inputs: those 13 engineered features plus the 19 raw pre-decision credit-history fields (credit limit, repayment status `PAY_*`, and the bill and payment amounts). Every one is a row-wise transform of pre-decision history, so there are no cross-row statistics, no target leakage, and no future information. The four protected demographic attributes — sex, age, education, and marital status — are **excluded from the model** and used only as axes for the fairness audit; the model never sees them. The same feature transforms are also written in SQL (`sql/feature_extraction.sql`) so the features can be built directly in a warehouse.

**Models.** A logistic-regression model as an interpretable baseline, and a HistGradientBoosting model as the challenger. The baseline is kept as a fixed interpretable reference (not hyperparameter-tuned) — standard champion-vs-challenger practice — and left at default class weighting so its probabilities stay calibrated and the Brier comparison is apples-to-apples. The challenger's probabilities are isotonic-calibrated, so a predicted PD can be read as an actual default rate.

**Validation.** I split off a 30% hold-out once and never touch it for fitting, tuning, or calibration. Hyperparameters come from 5-fold cross-validation on the training set, and I report both the CV AUC and the train-vs-test gap so overfitting is something you can see rather than take on trust.

## Results (30% hold-out)

| Model | AUC | Gini | KS | Brier | PSI |
|---|--:|--:|--:|--:|--:|
| Logistic regression (baseline) | 0.7495 | 0.4991 | 0.3961 | 0.1427 | 0.0007 |
| **HistGradientBoosting + isotonic** | **0.7814** | **0.5627** | **0.4278** | **0.1348** | **0.0024** |

Both models are calibrated, so the Brier comparison is apples-to-apples: the champion (0.1348) edges the baseline (0.1427) while leading clearly on discrimination (AUC/Gini/KS). 5-fold cross-validated AUC on the training set is 0.7869, effectively equal to the 0.7814 hold-out. That CV-to-test gap of about 0.006 is the real evidence the model generalizes; the raw train-vs-test gap (0.0311) looks larger only because it compares training-set resubstitution to test. Worth saying: on a dataset this well studied, an AUC much above 0.80 usually means something has leaked, so a credible 0.78 was the goal, not a bigger headline number.

(The PSI column above is a train-vs-test score-distribution check on a single random split, near zero by construction — not an out-of-time drift metric. Real drift monitoring is in the roadmap.)

![Champion model hold-out performance: ROC, precision-recall, calibration, score separation](outputs/model_performance.png)

## Explainability

Global drivers come from permutation importance and SHAP, with the logistic coefficients as an interpretable cross-check. The two dominant drivers are delinquency signals: the worst delinquency in the last six months (`pay_max`) ranks highest, followed by the most recent repayment status (`PAY_1`); utilization (`util_max`, `util_recent`) and balance trend (`bill_trend`) follow well behind. These global rankings are computed on the uncalibrated base of the isotonic champion, which is a monotone transform of it, so per-applicant driver rankings are unchanged. Per-applicant SHAP values feed the adverse-action layer below.

![SHAP summary](outputs/shap_summary.png)

## Where it breaks (`outputs/findings.md`)

I spent as much time on where the model fails as on its headline number:
- Calibration is weakest in the high-risk tail, which is exactly where declines happen, so those cases should go to a person rather than an automatic decline.
- Discrimination is not uniform across segments, so one global cutoff is not equally fair or accurate for everyone.
- About 174 accounts scored below 10% PD still default (6% of that group). These confident false negatives are the quiet, expensive errors that pure auto-approval would wave through.

## Fairness & governance

The four protected attributes — sex, age, education, and marital status — are excluded from the model itself and audited only for **disparate impact**: the correct governance setup, where the model never sees a demographic yet its decisions are still sliced across them (`outputs/fairness.csv`). Decline-rate parity clears the 4/5ths rule on sex (0.882), marital status (0.91), and, marginally, age (0.801), but falls below it on education (0.423) — a figure driven mostly by a tiny n=120 "other" group with an unusually low decline rate, whose per-group AUC (0.616) is itself unreliable. I flag the education disparity as a fairness risk rather than smooth it over. An example of the constrained adverse-action output is in [`outputs/reason_codes_example.md`](outputs/reason_codes_example.md) (template fallback; set `ANTHROPIC_API_KEY` for the LLM version). [`MODEL_CARD.md`](MODEL_CARD.md) records intended use, limitations, monitoring (PSI and calibration drift), human checkpoints, and how the work lines up with **OSFI E-23** and **FCAC**. The LLM adverse-action layer is deliberately constrained: it can cite only the model's actual top risk drivers for an applicant, so it cannot invent a reason the model did not use.

## Roadmap / next steps

These are the steps that would take this work sample from a point-in-time demo toward a model an origination team could actually deploy and defend.

**Validation & monitoring**
- Run a true out-of-time / vintage split instead of the random stratified hold-out: train on earlier origination months, test on later ones to expose temporal drift. Pair it with a feature-staleness ablation on `pay_max` and `PAY_1` (the two dominant drivers) to bound refresh cadence. This is the single largest gap and should come first.
- Replace the monitoring prose with an executable drift monitor: persist a versioned training reference (score bins, per-feature bins, calibration curve) and score each new batch for feature PSI, score PSI, and a calibration-drift test (Spiegelhalter Z / ECE), emitting warn/fail at the documented thresholds. Current PSI is train-vs-test from a single split and is near zero by construction.
- Gate champion-challenger promotion in CI (minimum AUC lift, maximum overfit gap, maximum ECE, minimum fairness parity), and serialize the fitted pipeline, data checksum, pinned package versions, and seed to a run manifest, so any model swap is auditable and reproducible.

**Modeling**
- Fix the high-risk tail: benchmark isotonic against a monotone beta / Platt recalibration on a dedicated calibration split, bootstrap the calibration map for per-decile confidence bands, and map PD to a rating master scale. Isotonic is high-variance where the top decile thins out (0.717 predicted vs 0.696 observed).
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

**Reproducibility.** Run on Python 3.9 with a fixed seed (`RNG = 42`) and a single 30% stratified hold-out split; the pipeline finishes in roughly 20–30 s on a modern laptop. Dependencies are pinned in `requirements.txt`, and the exact figures above are for that tested stack.

## Structure
```
src/pipeline.py         model build, validation, fairness, explainability, failure-mode analysis
src/features.py         shared leakage-safe feature engineering
sql/feature_extraction.sql  the same feature logic in SQL, against a warehouse schema
src/ai_reason_codes.py  LLM adverse-action layer, grounded in the model's real SHAP drivers
outputs/                metrics.json, fairness.csv, findings.md, plots
outputs/reason_codes_example.md  example constrained adverse-action notices (template fallback; set ANTHROPIC_API_KEY for the LLM version)
MODEL_CARD.md           intended use, performance, limitations, monitoring, governance
```
