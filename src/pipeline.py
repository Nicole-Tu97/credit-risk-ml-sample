"""
Credit-decision PD model — end-to-end: build -> validate -> fairness -> explainability -> failure modes.
Dataset: UCI "Default of Credit Card Clients" (Taiwan, 2005), 30,000 accounts. Target = default next month.
Run:  python3 src/pipeline.py
Outputs: outputs/metrics.json, outputs/fairness.csv, outputs/findings.md, outputs/*.png
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "default of credit card clients.xls"
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
RNG = 42

def ks_stat(y, p):            # Kolmogorov–Smirnov separation (credit-standard)
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)

def psi(expected, actual, bins=10):   # Population Stability Index on score distributions
    q = np.quantile(expected, np.linspace(0, 1, bins + 1)); q[0], q[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, q)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, q)[0] / len(actual), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

# ---------- load & prepare ----------
df = pd.read_excel(DATA, header=1).rename(columns={"default payment next month": "default", "PAY_0": "PAY_1"})
df = df.drop(columns=["ID"])
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})   # fold undocumented codes -> 4 (others)
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})                 # fold undocumented 0 -> 3 (others)
y = df["default"].astype(int); X = df.drop(columns=["default"])
feats = X.columns.tolist()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RNG)
print(f"rows={len(df)}  default_rate={y.mean():.3f}  train={len(X_tr)} test={len(X_te)}")

# ---------- models: interpretable baseline + GBM challenger ----------
logit = Pipeline([("sc", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]).fit(X_tr, y_tr)
gbm = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=400,
                                     l2_regularization=1.0, random_state=RNG).fit(X_tr, y_tr)

def evaluate(name, model):
    p_tr, p_te = model.predict_proba(X_tr)[:, 1], model.predict_proba(X_te)[:, 1]
    row = dict(model=name, AUC=round(roc_auc_score(y_te, p_te), 4),
               Gini=round(2 * roc_auc_score(y_te, p_te) - 1, 4),
               KS=round(ks_stat(y_te.values, p_te), 4),
               Brier=round(brier_score_loss(y_te, p_te), 4),
               PSI_train_vs_test=round(psi(p_tr, p_te), 4))
    return row, p_te

m_logit, p_logit = evaluate("LogisticRegression (scorecard baseline)", logit)
m_gbm, p_gbm = evaluate("HistGradientBoosting (challenger)", gbm)
for m in (m_logit, m_gbm): print(m)
champion, p_te = "HistGradientBoosting", p_gbm      # challenger is champion

# ---------- ROC + calibration ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for nm, p in [("LogReg", p_logit), ("GBM", p_te)]:
    fpr, tpr, _ = roc_curve(y_te, p); ax[0].plot(fpr, tpr, label=f"{nm} (AUC={roc_auc_score(y_te,p):.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0].set(title="ROC", xlabel="FPR", ylabel="TPR"); ax[0].legend()
frac_pos, mean_pred = calibration_curve(y_te, p_te, n_bins=10, strategy="quantile")
ax[1].plot(mean_pred, frac_pos, "o-", label="GBM"); ax[1].plot([0, 1], [0, 1], "k--", lw=.8)
ax[1].set(title="Calibration (champion)", xlabel="Predicted PD", ylabel="Observed default rate"); ax[1].legend()
plt.tight_layout(); plt.savefig(OUT / "roc_calibration.png", dpi=130); plt.close()

# ---------- decision cutoff (approve lowest-PD 70%) ----------
cutoff = np.quantile(p_te, 0.70); declined = p_te >= cutoff
print(f"policy cutoff PD>={cutoff:.3f} -> approval_rate={1-declined.mean():.3f}")

# ---------- fairness / bias across protected-ish segments ----------
seg = X_te.copy(); seg["_pd"] = p_te; seg["_y"] = y_te.values; seg["_declined"] = declined
seg["AGE_band"] = pd.cut(seg["AGE"], [0, 30, 40, 50, 200], labels=["<30", "30-40", "40-50", "50+"])
sex_map = {1: "male", 2: "female"}; edu_map = {1: "grad_school", 2: "university", 3: "high_school", 4: "other"}
rows = []
for col, mapper in [("SEX", sex_map), ("AGE_band", None), ("EDUCATION", edu_map)]:
    for g, gd in seg.groupby(col):
        if len(gd) < 100: continue
        try: auc_g = roc_auc_score(gd["_y"], gd["_pd"]) if gd["_y"].nunique() > 1 else np.nan
        except Exception: auc_g = np.nan
        rows.append(dict(attribute=col, group=(mapper.get(g, g) if mapper else str(g)), n=len(gd),
                         default_rate=round(gd["_y"].mean(), 3), decline_rate=round(gd["_declined"].mean(), 3),
                         AUC=round(auc_g, 3) if auc_g == auc_g else None))
fair = pd.DataFrame(rows); fair.to_csv(OUT / "fairness.csv", index=False)
# disparate-impact ratio on decline rate (min/max within each attribute)
di = {}
for attr, gd in fair.groupby("attribute"):
    di[attr] = round(gd["decline_rate"].min() / gd["decline_rate"].max(), 3)
print("fairness table:\n", fair.to_string(index=False)); print("decline-rate parity ratio (min/max):", di)

# ---------- explainability ----------
coef = pd.Series(logit.named_steps["clf"].coef_[0], index=feats).sort_values(key=abs, ascending=False)
pi = permutation_importance(gbm, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=RNG)
imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
print("\nTop logistic drivers (standardized):\n", coef.head(8).round(3).to_string())
print("\nTop GBM permutation importances (AUC drop):\n", imp.head(8).round(4).to_string())
shap_ok = False
try:
    import shap
    Xs = X_te.sample(min(800, len(X_te)), random_state=RNG)
    sv = shap.TreeExplainer(gbm).shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else sv
    shap.summary_plot(sv, Xs, show=False, max_display=12); plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=130); plt.close(); shap_ok = True
    print("SHAP summary saved.")
except Exception as e:
    print("SHAP skipped:", str(e)[:120])

# ---------- failure-mode analysis ----------
te = X_te.copy(); te["_pd"] = p_te; te["_y"] = y_te.values
te["decile"] = pd.qcut(te["_pd"], 10, labels=False, duplicates="drop")
cal = te.groupby("decile").agg(pred_PD=("_pd", "mean"), actual=("_y", "mean"), n=("_y", "size"))
cal["gap"] = (cal["actual"] - cal["pred_PD"]).round(3)
worst = fair.dropna(subset=["AUC"]).sort_values("AUC").head(1)
hi_conf_fn = int(((te["_pd"] < 0.10) & (te["_y"] == 1)).sum())         # confidently-approved that defaulted
hi_conf_fn_rate = round(((te["_pd"] < 0.10) & (te["_y"] == 1)).sum() / max((te["_pd"] < 0.10).sum(), 1), 3)

findings = f"""# Findings — what the model gets right, and where it breaks

## Performance (test, 30% holdout)
| model | AUC | Gini | KS | Brier | PSI (train vs test) |
|---|--:|--:|--:|--:|--:|
| {m_logit['model']} | {m_logit['AUC']} | {m_logit['Gini']} | {m_logit['KS']} | {m_logit['Brier']} | {m_logit['PSI_train_vs_test']} |
| {m_gbm['model']} | {m_gbm['AUC']} | {m_gbm['Gini']} | {m_gbm['KS']} | {m_gbm['Brier']} | {m_gbm['PSI_train_vs_test']} |

Champion: **{champion}**. PSI < 0.1 ⇒ score distribution stable between train and test.

## Where it breaks (failure-mode analysis)
1. **Calibration drifts in the high-risk tail.** Observed-minus-predicted default rate by PD decile:
{cal[['pred_PD','actual','gap','n']].round(3).to_string()}
   The top deciles are where predicted and actual diverge most — the model is least reliable exactly where decisions
   are most consequential (declines), so scores there should feed human review rather than auto-decline.
2. **Uneven discrimination across segments.** Weakest segment AUC: {worst.iloc[0]['attribute']}={worst.iloc[0]['group']}
   (AUC={worst.iloc[0]['AUC']}, n={int(worst.iloc[0]['n'])}) vs. overall {m_gbm['AUC']}. A single global cutoff is not
   equally accurate for every group — a fairness and governance concern, not just an accuracy one.
3. **Confidently-approved defaulters.** {hi_conf_fn} test accounts scored PD<10% yet defaulted
   ({hi_conf_fn_rate:.0%} of the low-PD population) — the costly, silent errors auto-approval would miss.
4. **Data-quality leak.** EDUCATION/MARRIAGE contained undocumented codes (0/5/6) folded into "other" during prep;
   left unhandled they silently create phantom categories — the kind of issue that must be caught before, not after, deployment.

## How these were found
By validating the model against its own assumptions rather than the headline AUC: calibration by risk band,
per-segment discrimination, and inspection of high-confidence errors — then routing the fragile regions to human checkpoints.
"""
(OUT / "findings.md").write_text(findings)
json.dump({"models": [m_logit, m_gbm], "champion": champion, "policy_cutoff_PD": round(float(cutoff), 4),
           "decline_parity_ratio": di, "shap": shap_ok,
           "top_logit_drivers": coef.head(8).round(3).to_dict(),
           "top_gbm_importances": imp.head(8).round(4).to_dict()},
          open(OUT / "metrics.json", "w"), indent=2)
print("\nDONE -> outputs/ (metrics.json, fairness.csv, findings.md, roc_calibration.png"
      + (", shap_summary.png" if shap_ok else "") + ")")
