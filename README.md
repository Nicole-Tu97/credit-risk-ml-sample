# Credit-Decision PD Model — a governed, AI-augmented workflow

A compact, end-to-end **probability-of-default (PD) model** for a lending decision, built the way a
defensible credit-risk function should be: not just a classifier, but **validation, fairness, explainability,
failure-mode analysis, and an AI-augmented adverse-action layer** — with the governance documented up front.

Built as a work sample for a *Modeling & Risk Management* role: it mirrors, in miniature, an
**AI-augmented credit-risk function** — models that drive portfolio decisions, the governance that keeps
them accurate and defensible, and the human/AI boundary around them.

> Data: **UCI "Default of Credit Card Clients"** (Taiwan, 2005; 30,000 accounts) — public, see `data/README.md`.
> Nothing here uses proprietary data.

## What it demonstrates (mapped to a credit-risk mandate)
| Capability | Where |
|---|---|
| Build **and validate** ML models (not just use them) | `src/pipeline.py` → LogReg baseline + GBM challenger |
| Credit-standard validation: **AUC, Gini, KS, calibration, PSI (stability)** | `outputs/metrics.json`, `roc_calibration.png` |
| **Fairness / bias** across segments (sex, age, education) | `outputs/fairness.csv` |
| **Explainability** (coefficients, permutation importance, **SHAP**) | `outputs/shap_summary.png`, `metrics.json` |
| **Failure-mode analysis** — where the model breaks and why | `outputs/findings.md` |
| **AI-augmented** adverse-action reason codes (LLM, grounded in real drivers) | `src/ai_reason_codes.py` → `outputs/reason_codes_example.md` |
| **Governance** — validation cadence, human checkpoints, OSFI E-23 / FCAC alignment | `MODEL_CARD.md` |

## Results (30% hold-out — untouched until final evaluation)
| Model | AUC | Gini | KS | Brier | PSI | train AUC | gap |
|---|--:|--:|--:|--:|--:|--:|--:|
| Logistic regression (scorecard baseline) | 0.754 | 0.509 | 0.397 | 0.192 | 0.001 | 0.765 | 0.010 |
| **HistGradientBoosting + isotonic (champion)** | **0.784** | **0.569** | **0.429** | **0.134** | **0.001** | 0.815 | 0.031 |

Leakage-safe feature engineering (23 → 36 features), hyperparameters CV-tuned on train, probabilities isotonic-calibrated inside train.
**Validity:** 5-fold CV AUC on train = **0.787** ≈ test AUC **0.784**; train–test gap **0.031**; PSI ≈ 0 — a credible,
well-validated lift, not an overfit or leaked one. (On this well-known dataset, an AUC far above ~0.80 would itself
be a red flag for leakage — so the goal was a *validated* result, not a vanity number.) Top driver: **recent repayment status (`PAY_1`)**.

**What it gets wrong (see `outputs/findings.md`):** calibration drifts in the high-risk tail (least reliable exactly
where declines happen → route to human review); discrimination is uneven across segments (a single global cutoff
isn't equally fair/accurate for every group); and a set of confidently-approved accounts still default — the silent,
costly errors that pure auto-approval would miss. These were found by validating the model **against its own
assumptions**, not the headline AUC.

## Run it
```bash
pip install -r requirements.txt
python3 src/download_data.py         # fetches the public UCI dataset into data/
python3 src/pipeline.py              # build → validate → fairness → explainability → failure modes
python3 src/ai_reason_codes.py       # AI adverse-action notices (set ANTHROPIC_API_KEY to use the LLM)
```

## Structure
```
src/pipeline.py          model build, validation, fairness, explainability, failure-mode analysis
src/ai_reason_codes.py   LLM/agentic adverse-action layer (grounded in the model's real SHAP drivers)
src/download_data.py     fetch the public dataset
outputs/                 metrics.json, fairness.csv, findings.md, plots, reason_codes_example.md
MODEL_CARD.md            intended use, performance, fairness, limitations, monitoring & regulatory alignment
```

## Governance note
The AI layer is deliberately constrained: it may cite **only** the model's actual top risk drivers for an
application — it cannot introduce factors the model did not use. Model risk, monitoring cadence, human
checkpoints, and alignment to **OSFI E-23** (model risk management) and **FCAC** (plain-language adverse action)
are documented in [`MODEL_CARD.md`](MODEL_CARD.md).
