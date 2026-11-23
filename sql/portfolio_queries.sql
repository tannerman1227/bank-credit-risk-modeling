-- Portfolio questions a credit analyst would run on the cleaned book.
-- The Python runner executes each block in SQLite against two tables:
--   accounts     full cleaned portfolio (monitoring, not used to fit the model)
--   scored_book  holdout accounts scored by the champion model
-- Amounts are New Taiwan dollars, the unit of the UCI file.

-- name: portfolio_snapshot
SELECT
    COUNT(*) AS accounts,
    SUM(default_flag) AS defaults,
    ROUND(AVG(default_flag), 4) AS default_rate,
    ROUND(AVG(limit_bal), 0) AS avg_limit,
    ROUND(AVG(CASE WHEN bill_amt1 > 0 THEN bill_amt1 END), 0) AS avg_positive_balance,
    ROUND(AVG(util_1), 4) AS avg_utilization
FROM accounts;

-- name: default_rate_by_repayment_status
SELECT
    pay_1 AS repayment_status,
    COUNT(*) AS accounts,
    SUM(default_flag) AS defaults,
    ROUND(AVG(default_flag), 4) AS default_rate,
    ROUND(AVG(limit_bal), 0) AS avg_limit,
    ROUND(AVG(bill_amt1), 0) AS avg_balance
FROM accounts
GROUP BY pay_1
ORDER BY pay_1;

-- name: default_rate_by_utilization
SELECT
    CASE
        WHEN util_1 < 0 THEN '1. Credit balance'
        WHEN util_1 < 0.30 THEN '2. 0-30%'
        WHEN util_1 < 0.70 THEN '3. 30-70%'
        WHEN util_1 < 1.00 THEN '4. 70-100%'
        ELSE '5. Overlimit'
    END AS utilization_band,
    COUNT(*) AS accounts,
    ROUND(AVG(default_flag), 4) AS default_rate,
    ROUND(AVG(limit_bal), 0) AS avg_limit,
    ROUND(AVG(CASE WHEN bill_amt1 > 0 THEN bill_amt1 END), 0) AS avg_positive_balance
FROM accounts
GROUP BY 1
ORDER BY 1;

-- name: default_rate_by_education
SELECT
    education,
    education_label,
    COUNT(*) AS accounts,
    ROUND(AVG(default_flag), 4) AS default_rate,
    ROUND(AVG(limit_bal), 0) AS avg_limit,
    ROUND(AVG(util_1), 4) AS avg_utilization
FROM accounts
GROUP BY education, education_label
ORDER BY education;

-- name: default_rate_by_age_band
SELECT
    CASE
        WHEN age <= 25 THEN '21-25'
        WHEN age <= 35 THEN '26-35'
        WHEN age <= 45 THEN '36-45'
        WHEN age <= 55 THEN '46-55'
        ELSE '56+'
    END AS age_band,
    COUNT(*) AS accounts,
    SUM(default_flag) AS defaults,
    ROUND(AVG(default_flag), 4) AS default_rate,
    ROUND(AVG(limit_bal), 0) AS avg_limit
FROM accounts
GROUP BY age_band
ORDER BY MIN(age);

-- name: behavior_segments
SELECT
    CASE
        WHEN pay_1 <= 0 AND util_1 >= 0.70 THEN 'Current but highly utilized'
        WHEN pay_1 <= 0 THEN 'Current, utilization under 70%'
        WHEN pay_1 = 1 THEN '1 month late'
        ELSE '2+ months late'
    END AS segment,
    COUNT(*) AS accounts,
    ROUND(AVG(default_flag), 4) AS default_rate,
    ROUND(AVG(util_1), 4) AS avg_utilization,
    ROUND(AVG(bill_amt1), 0) AS avg_balance
FROM accounts
GROUP BY segment
ORDER BY
    CASE segment
        WHEN 'Current, utilization under 70%' THEN 1
        WHEN 'Current but highly utilized' THEN 2
        WHEN '1 month late' THEN 3
        ELSE 4
    END;

-- name: delinquent_balance_watchlist
SELECT
    account_id,
    limit_bal,
    ROUND(bill_amt1, 0) AS bill_amt1,
    ROUND(util_1, 3) AS util_1,
    pay_1,
    months_late,
    default_flag
FROM accounts
WHERE pay_1 >= 2
  AND bill_amt1 >= 50000
ORDER BY bill_amt1 DESC
LIMIT 25;

-- name: expected_loss_by_risk_grade
SELECT
    risk_grade,
    COUNT(*) AS accounts,
    ROUND(AVG(pd), 4) AS avg_pd,
    ROUND(AVG(default_flag), 4) AS observed_default_rate,
    ROUND(SUM(ead), 0) AS total_ead,
    ROUND(SUM(expected_loss), 0) AS total_expected_loss,
    ROUND(SUM(expected_loss) / NULLIF(SUM(ead), 0), 4) AS el_rate
FROM scored_book
GROUP BY risk_grade
ORDER BY risk_grade;

-- name: expected_loss_concentration
WITH totals AS (
    SELECT SUM(expected_loss) AS total_el, SUM(ead) AS total_ead
    FROM scored_book
),
worst AS (
    SELECT SUM(expected_loss) AS worst_el
    FROM scored_book
    WHERE risk_grade = (SELECT MAX(risk_grade) FROM scored_book)
)
SELECT
    ROUND(totals.total_el, 0) AS total_expected_loss,
    ROUND(totals.total_ead, 0) AS total_ead,
    ROUND(totals.total_el / NULLIF(totals.total_ead, 0), 4) AS el_rate,
    ROUND(worst.worst_el, 0) AS highest_grade_expected_loss,
    ROUND(worst.worst_el / NULLIF(totals.total_el, 0), 4) AS share_of_el_in_highest_grade
FROM totals, worst;
