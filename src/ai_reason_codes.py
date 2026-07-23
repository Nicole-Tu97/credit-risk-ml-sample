"""
AI-augmented explainability layer: turn a model decision into a plain-language,
FCAC-style adverse-action notice grounded in the model's ACTUAL top risk drivers.

Governance choices:
  * The LLM is given only the applicant's top SHAP contributors and may explain ONLY those factors —
    it cannot introduce a reason the model did not use (no hallucinated factors).
  * Runs with ANTHROPIC_API_KEY set; otherwise a deterministic template fallback keeps it reproducible.

Run:  python3 src/ai_reason_codes.py   (add ANTHROPIC_API_KEY to use the LLM)
Out:  outputs/reason_codes_example.md
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from features import engineer, friendly

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "default of credit card clients.xls"
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
# champion-like config (mirrors the tuned params from pipeline.py)
PARAMS = dict(max_depth=5, learning_rate=0.02, max_iter=800, max_leaf_nodes=15,
              l2_regularization=5.0, early_stopping=True, validation_fraction=0.15, random_state=42)

def load_train():
    df = pd.read_excel(DATA, header=1).rename(columns={"default payment next month": "default", "PAY_0": "PAY_1"}).drop(columns=["ID"])
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4}); df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    y = df["default"].astype(int); X = df.drop(columns=["default"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    Xtr, Xte = engineer(Xtr), engineer(Xte)
    gbm = HistGradientBoostingClassifier(**PARAMS).fit(Xtr, ytr)
    return gbm, Xtr, Xte

def top_drivers(gbm, Xtr, Xte, row_idx, k=4):
    import shap
    expl = shap.TreeExplainer(gbm, Xtr.sample(200, random_state=1))
    sv = expl.shap_values(Xte.iloc[[row_idx]]); sv = (sv[1] if isinstance(sv, list) else sv)[0]
    adverse = pd.Series(sv, index=Xte.columns)
    adverse = adverse[adverse > 0].sort_values(ascending=False).head(k)      # factors pushing risk UP
    return [(f, friendly(f), round(float(Xte.iloc[row_idx][f]), 2)) for f in adverse.index]

def llm_notice(pd_score, drivers):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    factors = "; ".join(f"{d[1]} (value={d[2]})" for d in drivers)
    try:
        import anthropic
        prompt = (
            "You are a credit adjudication assistant writing an adverse-action notice for a declined small-business "
            "loan applicant, in clear plain language consistent with Canadian FCAC expectations.\n"
            f"Model probability of default: {pd_score:.0%}.\n"
            f"The ONLY risk factors you may cite (from the model's own explanation) are: {factors}.\n"
            "Write 3-4 short, respectful bullet reason codes. Do NOT invent any factor not listed. "
            "Do NOT give financial advice or guarantees. End with one neutral sentence on requesting a human review."
        )
        m = anthropic.Anthropic().messages.create(model="claude-sonnet-5", max_tokens=400,
                                                  messages=[{"role": "user", "content": prompt}])
        return m.content[0].text
    except Exception as e:
        return f"[LLM call failed: {str(e)[:80]} — template fallback shown]"

def template_notice(pd_score, drivers):
    lines = [f"Decision: application declined (model-estimated probability of default: {pd_score:.0%}).",
             "Principal reasons (based only on the model's top risk drivers for this application):"]
    lines += [f"  • {desc.capitalize()} (recorded value: {val}) weighed against approval." for _, desc, val in drivers]
    lines.append("You may request a manual review; a human adjudicator will re-assess your file.")
    return "\n".join(lines)

if __name__ == "__main__":
    gbm, Xtr, Xte = load_train()
    p = gbm.predict_proba(Xte)[:, 1]
    out = ["# AI-generated adverse-action notices (example)\n",
           "_Reason codes are grounded in each applicant's actual top SHAP risk drivers; the LLM may cite only those factors._\n"]
    for i in np.argsort(p)[-3:][::-1]:                       # three highest-risk (declined) applicants
        drivers = top_drivers(gbm, Xtr, Xte, int(i))
        notice = llm_notice(float(p[i]), drivers)
        mode = "LLM (claude-sonnet-5)" if (notice and not notice.startswith("[LLM")) else "template fallback"
        notice = notice or template_notice(float(p[i]), drivers)
        out += [f"## Applicant #{int(i)} — PD={p[i]:.0%}  _(generated via {mode})_", "```", notice, "```", ""]
    (OUT / "reason_codes_example.md").write_text("\n".join(out))
    print("wrote outputs/reason_codes_example.md  (set ANTHROPIC_API_KEY to generate via the LLM)")
