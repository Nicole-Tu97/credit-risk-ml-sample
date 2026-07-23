-- Feature extraction for the PD model, expressed in SQL against a warehouse schema.
-- This mirrors the row-wise transforms in ../src/features.py, so the exact same features
-- can be built directly in a data warehouse (Redshift / Postgres / BigQuery-style SQL)
-- instead of in pandas. Every input is pre-decision history: no target, no future data.
--
-- Assumed (normalized) schema:
--   accounts(client_id, limit_bal, sex, education, marriage, age)
--   monthly_statements(client_id, months_ago, pay_status, bill_amt, pay_amt)   -- months_ago = 1..6
--
-- Output: one row per client with the model's engineered features.

WITH stmt AS (
    SELECT
        client_id,
        MAX(pay_status)                                              AS pay_max,        -- worst delinquency in 6 months
        AVG(pay_status)                                              AS pay_mean,       -- average delinquency
        SUM(CASE WHEN pay_status > 0 THEN 1 ELSE 0 END)             AS n_delinq,       -- # delinquent months
        MAX(CASE WHEN months_ago = 1 THEN pay_status END)          AS pay_1,          -- most recent status
        AVG(bill_amt)                                               AS bill_mean,
        MAX(bill_amt)                                               AS bill_max,
        MAX(CASE WHEN months_ago = 1 THEN bill_amt END)            AS bill_recent,
        MAX(CASE WHEN months_ago = 6 THEN bill_amt END)            AS bill_oldest,
        SUM(bill_amt)                                               AS bill_sum,
        SUM(pay_amt)                                                AS pay_amt_sum,     -- total paid over 6 months
        SUM(CASE WHEN pay_amt = 0 THEN 1 ELSE 0 END)               AS n_zero_pay       -- months with no payment
    FROM monthly_statements
    WHERE months_ago BETWEEN 1 AND 6
    GROUP BY client_id
)
SELECT
    a.client_id,
    a.limit_bal,
    -- utilization (balance / assigned limit); NULLIF guards a zero limit
    s.bill_recent / NULLIF(a.limit_bal, 0)                          AS util_recent,
    s.bill_mean   / NULLIF(a.limit_bal, 0)                          AS util_avg,
    s.bill_max    / NULLIF(a.limit_bal, 0)                          AS util_max,
    -- delinquency
    s.pay_max,
    s.pay_mean,
    s.n_delinq,
    CASE WHEN s.pay_1 > 0 THEN 1 ELSE 0 END                        AS curr_delinq,
    s.pay_1 - s.pay_mean                                            AS delinq_trend,
    -- payment behaviour
    s.pay_amt_sum,
    s.n_zero_pay,
    s.pay_amt_sum / (ABS(s.bill_sum) + 1)                          AS pay_to_bill,
    -- balance level and trend
    s.bill_mean,
    s.bill_recent - s.bill_oldest                                   AS bill_trend
FROM accounts a
JOIN stmt s USING (client_id);
