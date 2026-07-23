"""
AI-augmented explainability layer: turn a model decision into a plain-language,
FCAC-style adverse-action notice grounded in the model's ACTUAL top risk drivers.

Design choices that matter for governance:
  * The LLM never sees raw model internals or invents reasons — it is given the
    applicant's top SHAP contributors and must explain ONLY those (no hallucinated factors).
  * Runs with ANTHROPIC_API_KEY set; otherwise falls back to a deterministic template,
    so the pipeline is reproducible without a key.

Run:  python3 src/ai_reason_codes.py   (add ANTHROPIC_API_KEY to use the LLM)
Out:  outputs/reason_codes_example.md
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "default of credit card clients.xls"
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)

FRIENDLY = {  # model feature -> applicant-facing description
    "PAY_1": "most recent repayment status (months delinquent)",
    "PAY_2": "repayment status 2 months ago", "PAY_3": "repayment status 3 months ago",
    "PAY_4": "repayment status 4 months ago", "PAY_5": "repayment status 5 months ago",
    "PAY_6": "repayment status 6 months ago",
    "LIMIT_BAL": "assigned credit limit", "BILL_AMT1": "most recent statement balance",
    "BILL_AMT2": "statement balance 2 months ago", "BILL_AMT3": "statement balance 3 months ago",
    "PAY_AMT1": "most recent payment amount", "PAY_AMT2": "prior payment amount",
    "PAY_AMT3": "payment amount 3 months ago",
    "AGE": "age", "EDUCATION": "education level", "MARRIAGE": "marital status",
}
def friendly(f):
    return FRIENDLY.get(f, f.replace("_", " ").lower())

def load_train():
    df = pd.read_excel(DATA, header=1).rename(columns={"default payment next month": "default", "PAY_0": "PAY_1"}).drop(columns=["ID"])
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4}); df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    y = df["default"].astype(int); X = df.drop(columns=["default"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    gbm = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=400,
                                         l2_regularization=1.0, random_state=42).fit(Xtr, ytr)
    return gbm, Xtr, Xte

def top_drivers(gbm, Xtr, Xte, row_idx, k=4):
    import shap
    expl = shap.TreeExplainer(gbm, Xtr.sample(200, random_state=1))
    sv = expl.shap_values(Xte.iloc[[row_idx]])
    sv = (sv[1] if isinstance(sv, list) else sv)[0]
    s = pd.Series(sv, index=Xte.columns)
    adverse = s[s > 0].sort_values(ascending=False).head(k)   # factors pushing risk UP
    return [(f, friendly(f), round(float(Xte.iloc[row_idx][f]), 2)) for f in adverse.index]

def llm_notice(pd_score, drivers):
    key = os.environ.get("ANTHROPIC_API_KEY")
    factors = "; ".join(f"{d[1]} (value={d[2]})" for d in drivers)
    if not key:
        return None
    try:
        import anthropic
        prompt = (
            "You are a credit adjudication assistant writing an adverse-action notice for a declined small-business "
            "loan applicant, compliant with Canadian FCAC plain-language expectations.\n"
            f"Model probability of default: {pd_score:.0%}.\n"
            f"The ONLY approved risk factors you may cite (from the model's explanation) are: {factors}.\n"
            "Write 3-4 short bullet reason codes in plain, respectful language. Do NOT invent any factor not listed. "
            "Do NOT give financial advice or guarantees. End with one neutral sentence on how to request a review."
        )
        msg = anthropic.Anthropic(api_key=key).messages.create(
            model="claude-sonnet-5", max_tokens=400, messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    except Exception as e:
        return f"[LLM call failed: {str(e)[:80]} — using template fallback below]"

def template_notice(pd_score, drivers):
    lines = [f"Decision: application declined (model-estimated probability of default: {pd_score:.0%}).",
             "Principal reasons (based only on the model's top risk drivers for this application):"]
    for _, desc, val in drivers:
        lines.append(f"  • {desc.capitalize()} (recorded value: {val}) weighed against approval.")
    lines.append("You may request a manual review of this decision; a human adjudicator will re-assess your file.")
    return "\n".join(lines)

if __name__ == "__main__":
    gbm, Xtr, Xte = load_train()
    p = gbm.predict_proba(Xte)[:, 1]
    declined_idx = np.argsort(p)[-3:][::-1]          # three highest-risk (declined) applicants
    out = ["# AI-generated adverse-action notices (example)\n",
           "_Reason codes are grounded in each applicant's actual top SHAP risk drivers; the LLM may cite only those factors._\n"]
    for i in declined_idx:
        drivers = top_drivers(gbm, Xtr, Xte, int(i))
        notice = llm_notice(float(p[i]), drivers) or template_notice(float(p[i]), drivers)
        mode = "LLM (claude-sonnet-5)" if os.environ.get("ANTHROPIC_API_KEY") and not notice.startswith("[LLM") else "template fallback"
        out += [f"## Applicant #{int(i)} — PD={p[i]:.0%}  _(generated via {mode})_", "```", notice, "```", ""]
    (OUT / "reason_codes_example.md").write_text("\n".join(out))
    print("wrote outputs/reason_codes_example.md  (set ANTHROPIC_API_KEY to generate via the LLM)")
