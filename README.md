# Credit Default Prediction — a validated, governed PD model

A probability-of-default (PD) model on the UCI credit-card dataset, built end to end the way a
credit-risk function needs before a model ships: leakage-safe features, cross-validated tuning,
calibrated probabilities, and explicit validation, fairness, explainability, and failure-mode checks.

> Data: **UCI "Default of Credit Card Clients"** (Taiwan, 2005; 30,000 accounts). Public — nothing proprietary.

## Data
30,000 accounts, 23 raw features: credit limit, demographics, and six months of repayment status,
bill amounts, and payments. Target is default in the following month (22.1% base rate). Undocumented
`EDUCATION`/`MARRIAGE` codes are folded into "other" as a logged data-quality step.

## Approach
- **Features** — 23 → 36 leakage-safe features (utilization, delinquency streaks, payment-to-bill
  ratios, balance/delinquency trends). Every feature is a per-row transform of pre-decision history:
  no cross-row statistics, no target, no future information.
- **Models** — a logistic-regression scorecard baseline and a HistGradientBoosting challenger, with
  probabilities isotonic-calibrated.
- **Validation** — a 30% hold-out is split off once and never used for fitting, tuning, or calibration.
  Hyperparameters are tuned by 5-fold CV on the training set only, and CV AUC plus the train–test gap
  are reported so overfitting is measured rather than assumed away.

## Results (30% hold-out)

| Model | AUC | Gini | KS | Brier | PSI |
|---|--:|--:|--:|--:|--:|
| Logistic regression (baseline) | 0.754 | 0.509 | 0.397 | 0.192 | 0.001 |
| **HistGradientBoosting + isotonic** | **0.784** | **0.569** | **0.429** | **0.134** | **0.001** |

5-fold CV AUC on train is 0.787, in line with the 0.784 hold-out (train–test gap 0.031, PSI ≈ 0), so
the lift is validated rather than overfit. On this well-studied dataset an AUC far above ~0.80 usually
points to leakage, so a credible 0.78 was the target, not a higher vanity number.

![Champion model hold-out performance: ROC, precision-recall, calibration, score separation](outputs/model_performance.png)

## Explainability

Global importance from logistic coefficients, permutation importance, and SHAP. The dominant driver is
the most recent repayment status (`PAY_1`). Per-applicant SHAP values feed the adverse-action layer.

![SHAP summary](outputs/shap_summary.png)

## Where it breaks (`outputs/findings.md`)
- Calibration is weakest in the high-risk tail, exactly where declines happen — route those to human
  review rather than auto-decline.
- Discrimination is uneven across segments, so a single global cutoff is not equally fair or accurate.
- A set of confidently-approved accounts (PD < 10%) still default: the silent false negatives that
  pure auto-approval would miss.

## Fairness & governance
Discrimination and decline rates are computed across sex, age band, and education (`outputs/fairness.csv`);
parity is reasonable on sex and age but weak on education, which is flagged as a fairness risk.
[`MODEL_CARD.md`](MODEL_CARD.md) covers intended use, limitations, monitoring (PSI and calibration drift),
human checkpoints, and alignment to **OSFI E-23** and **FCAC**. The LLM adverse-action layer may cite only
the model's actual top risk drivers for a given applicant — it cannot introduce factors the model did not use.

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
src/ai_reason_codes.py  LLM adverse-action layer, grounded in the model's real SHAP drivers
outputs/                metrics.json, fairness.csv, findings.md, plots
MODEL_CARD.md           intended use, performance, limitations, monitoring, governance
```
