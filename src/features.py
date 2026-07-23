"""Shared, leakage-safe feature engineering (imported by pipeline.py and ai_reason_codes.py).

All engineered features are deterministic per-row transforms of historical, pre-decision columns
(repayment status PAY_*, statement balances BILL_AMT*, payments PAY_AMT*). No cross-row statistics,
no target, no future information -> safe to apply to train and test independently.
"""
import numpy as np

PAYS = [f"PAY_{i}" for i in (1, 2, 3, 4, 5, 6)]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
AMTS = [f"PAY_AMT{i}" for i in range(1, 7)]

FRIENDLY = {  # model feature -> applicant-facing description (for adverse-action reason codes)
    "PAY_1": "most recent repayment status (months delinquent)",
    "PAY_2": "repayment status 2 months ago", "PAY_3": "repayment status 3 months ago",
    "PAY_4": "repayment status 4 months ago", "PAY_5": "repayment status 5 months ago",
    "PAY_6": "repayment status 6 months ago",
    "LIMIT_BAL": "assigned credit limit", "BILL_AMT1": "most recent statement balance",
    "PAY_AMT1": "most recent payment amount", "PAY_AMT2": "prior payment amount",
    "util_recent": "recent credit utilization (balance ÷ limit)", "util_avg": "average credit utilization",
    "util_max": "peak credit utilization", "pay_max": "worst delinquency in the last 6 months",
    "pay_mean": "average delinquency over 6 months", "n_delinq": "number of delinquent months (of 6)",
    "curr_delinq": "currently delinquent", "delinq_trend": "recent vs. average delinquency trend",
    "pay_to_bill": "repayment coverage (payments ÷ billed)", "n_zero_pay": "months with no payment",
    "bill_trend": "6-month change in statement balance", "pay_amt_sum": "total payments over the last 6 months",
    "bill_mean": "average statement balance",
}

def friendly(f):
    return FRIENDLY.get(f, f.replace("_", " ").lower())

def engineer(df):
    d = df.copy(); lim = d["LIMIT_BAL"].replace(0, np.nan)
    d["util_recent"] = d["BILL_AMT1"] / lim
    d["util_avg"] = d[BILLS].mean(axis=1) / lim
    d["util_max"] = d[BILLS].max(axis=1) / lim
    d["pay_max"] = d[PAYS].max(axis=1)
    d["pay_mean"] = d[PAYS].mean(axis=1)
    d["n_delinq"] = (d[PAYS] > 0).sum(axis=1)
    d["curr_delinq"] = (d["PAY_1"] > 0).astype(int)
    d["delinq_trend"] = d["PAY_1"] - d[PAYS].mean(axis=1)
    d["pay_amt_sum"] = d[AMTS].sum(axis=1)
    d["n_zero_pay"] = (d[AMTS] == 0).sum(axis=1)
    d["pay_to_bill"] = d[AMTS].sum(axis=1) / (d[BILLS].sum(axis=1).abs() + 1)
    d["bill_mean"] = d[BILLS].mean(axis=1)
    d["bill_trend"] = d["BILL_AMT1"] - d["BILL_AMT6"]
    return d.replace([np.inf, -np.inf], np.nan).fillna(0.0)
