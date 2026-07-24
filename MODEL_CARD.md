# Model Card — Consumer Revolving-Credit PD Model (work sample)

## Intended use
Estimate an applicant's **probability of default (PD)** to support a lending decision. Intended as a
**decision-support** input feeding a policy cutoff **plus human review**, not a fully automated approve/decline.

**Not for:** final automated adverse decisions without human accountability; use on populations materially
different from the training data; any purpose where the documented limitations below are not monitored.

## Data
UCI *Default of Credit Card Clients* (Taiwan, 2005), 30,000 accounts, 23 features (limit, demographics,
6-month repayment status / bill / payment history). Target: default in the next month (base rate 22.1%).
Preprocessing: undocumented `EDUCATION`/`MARRIAGE` category codes (0/5/6) folded into "other" — an explicit
data-quality control, logged rather than left to create phantom categories.

## Models & selection
Interpretable **logistic-regression** baseline vs. **HistGradientBoosting** challenger, trained on 32 leakage-safe
inputs (13 engineered + 19 raw credit-history fields). **Protected demographic attributes (sex, age, education,
marital status) are excluded from the model** and used only for the disparate-impact audit. Champion = GBM on
discrimination and calibration (see README table). The baseline is a fixed interpretable reference (not
hyperparameter-tuned) retained for the challenger–champion comparison; it is left at default class weighting so its
probabilities stay calibrated and the Brier comparison against the isotonic champion is apples-to-apples.

## Performance (30% stratified hold-out)
Champion (HistGradientBoosting + isotonic): **AUC 0.7814 · Gini 0.5627 · KS 0.4278 · Brier 0.1348** (baseline: AUC 0.7495 · Brier 0.1427).
**Validity:** 5-fold CV AUC on train 0.7869 ≈ test 0.7814 — a CV-to-test gap of ~0.006, which is the real generalization
check; the raw train-vs-test gap of 0.0311 is resubstitution-vs-test and naturally larger. 32 leakage-safe model
features (13 engineered + 19 raw credit-history), CV-tuned, isotonic-calibrated inside train. (PSI in the outputs is a
single-split train/test score-distribution check, ≈0 by construction — not an out-of-time drift metric; true
out-of-time validation is the top roadmap item.)
Full numbers in `outputs/metrics.json`; ROC, precision-recall, calibration & score separation in `outputs/model_performance.png`.

## Fairness assessment
The model never sees protected attributes; they are used **only** for disparate-impact testing. Discrimination and
decline rates are computed across **sex, age band, education, and marital status** (`outputs/fairness.csv`).
Findings: decline-rate parity clears the 4/5ths rule on sex (0.882), marital status (0.91), and — marginally — age
(0.801), but **falls below it on education (0.423)**, driven largely by a tiny "other" group (n=120) with an unusually
low decline rate, whose AUC (0.616) is itself unreliable. **Mitigation:** treat the education disparity as a fairness
risk, avoid a single global cutoff for thin/low-AUC segments, and route them to review.

## Explainability
Global: GBM permutation importance + SHAP summary, with logistic coefficients as a cross-check (`outputs/shap_summary.png`);
the top drivers are `pay_max` (worst delinquency in 6 months) and `PAY_1` (most recent repayment status). These global
rankings are computed on the uncalibrated base of the isotonic champion, which is monotone-equivalent, so per-applicant
driver rankings are unchanged.
Local: per-applicant SHAP drivers feed the adverse-action layer, which may cite **only** those factors.

## Limitations & failure modes (`outputs/findings.md`)
1. **Tail miscalibration** — predicted vs. observed default rate diverges most in the high-risk deciles.
2. **Uneven segment discrimination** — a global cutoff is not equally accurate/fair for every group.
3. **Confident false negatives** — some low-PD accounts still default (silent auto-approval risk).
4. **Point-in-time data** — no true out-of-time split available; temporal validation is a deployment prerequisite.

## Monitoring, validation & governance
- **Ongoing monitoring:** PSI on score & key features (recalibrate if PSI > 0.1–0.25); calibration and AUC/KS
  tracked on rolling vintages; alert on drift.
- **Validation cadence:** independent pre-deployment validation, then periodic revalidation; challenger models
  benchmarked before any champion swap.
- **Human checkpoints:** high-risk-tail and thin/low-AUC segments routed to human adjudication; adverse actions
  reviewable on request.
- **Regulatory alignment:** structured to **OSFI E-23** (model risk management — inventory, validation, monitoring,
  accountable owner) and **FCAC** (clear, plain-language adverse-action reasons; no hidden factors).

*Author: Yejun (Nicole) Tu. Public data; illustrative work sample, not a production model.*
