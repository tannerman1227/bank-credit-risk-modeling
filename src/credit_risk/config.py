"""Fixed modeling choices. These are not tuned on the holdout."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_PATH = DATA_DIR / "sample" / "credit_card_sample.csv"
RAW_XLS = RAW_DIR / "default_of_credit_card_clients.xls"
RAW_CSV = RAW_DIR / "default_of_credit_card_clients.csv"
FIGURE_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"
REPORT_DIR = ROOT / "reports"
SQL_PATH = ROOT / "sql" / "portfolio_queries.sql"
EXCEL_PATH = REPORT_DIR / "CreditRiskPortfolio.xlsx"
METRICS_PATH = REPORT_DIR / "metrics.json"

UCI_XLS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)

SEED = 42
TEST_SIZE = 0.20
N_SPLITS = 5
SAMPLE_ROWS = 250

# Prefer the interpretable scorecard unless boosting wins by at least this
# much cross-validated AUC. Half a point of AUC is the indifference margin.
CHAMPION_AUC_MARGIN = 0.005
# Use class weights only when they improve cross-validated AUC by this much.
WEIGHT_AUC_MARGIN = 0.005
# Drop a scorecard characteristic below this information value.
IV_MIN = 0.02

# Points-to-double-odds scaling. A linear transform of log-odds; ranking is unchanged.
PDO = 20
TARGET_SCORE = 600
TARGET_ODDS = 20.0  # goods per bad at the target score

# The file has no recoveries and no drawn-at-default history.
# These are illustrative assumptions so expected loss can be stated in currency.
# Primary severity is a high unsecured-card LGD. The 45% case is a sensitivity.
ASSUMED_LGD = 0.75
LGD_SENSITIVITIES = (0.75, 0.45)
ASSUMED_CCF = 0.10

# Share of the holdout kept if accounts are accepted from safest to riskiest.
POLICY_APPROVAL_RATE = 0.70
APPROVAL_GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)

N_GRADES = 5
N_DECILES = 10

EDUCATION_LABELS = {
    1: "Graduate school",
    2: "University",
    3: "High school",
    4: "Other",
}
MARRIAGE_LABELS = {1: "Married", 2: "Single", 3: "Other"}
SEX_LABELS = {1: "Male", 2: "Female"}

# Excluded from every model. Retained only for monitoring.
PROTECTED_COLUMNS = ("sex", "sex_label", "marriage", "marriage_label")

SCORECARD_FEATURES = [
    "pay_1",
    "months_late",
    "util_1",
    "pay_ratio_1",
    "limit_bal",
    "age",
    "education",
]

GBM_FEATURES = [
    "limit_bal",
    "age",
    "education",
    "pay_1",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6",
    "bill_amt1",
    "bill_amt2",
    "bill_amt3",
    "bill_amt4",
    "bill_amt5",
    "bill_amt6",
    "pay_amt1",
    "pay_amt2",
    "pay_amt3",
    "pay_amt4",
    "pay_amt5",
    "pay_amt6",
    "util_1",
    "util_2",
    "util_3",
    "util_4",
    "util_5",
    "util_6",
    "pay_ratio_1",
    "pay_ratio_2",
    "pay_ratio_3",
    "pay_ratio_4",
    "pay_ratio_5",
    "pay_ratio_6",
    "max_delay",
    "months_late",
    "months_severe",
    "avg_util",
    "util_recent_minus_old",
    "avg_pay_ratio",
]

GBM_PARAMS = {
    "n_estimators": 250,
    "learning_rate": 0.06,
    "max_depth": 3,
    "min_child_weight": 20,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": 1,
}

FEATURE_LABELS = {
    "pay_1": "Most recent repayment status",
    "pay_2": "Repayment status, 2 months prior",
    "pay_3": "Repayment status, 3 months prior",
    "pay_4": "Repayment status, 4 months prior",
    "pay_5": "Repayment status, 5 months prior",
    "pay_6": "Repayment status, 6 months prior",
    "months_late": "Count of late statements (of 6)",
    "months_severe": "Count of statements 2+ months late",
    "max_delay": "Worst repayment status in 6 months",
    "util_1": "Current utilization",
    "util_2": "Utilization, 2 months prior",
    "util_3": "Utilization, 3 months prior",
    "util_4": "Utilization, 4 months prior",
    "util_5": "Utilization, 5 months prior",
    "util_6": "Utilization, 6 months prior",
    "avg_util": "Average utilization",
    "util_recent_minus_old": "Utilization change (recent minus oldest)",
    "pay_ratio_1": "Current payment relative to bill",
    "pay_ratio_2": "Payment ratio, 2 months prior",
    "pay_ratio_3": "Payment ratio, 3 months prior",
    "pay_ratio_4": "Payment ratio, 4 months prior",
    "pay_ratio_5": "Payment ratio, 5 months prior",
    "pay_ratio_6": "Payment ratio, 6 months prior",
    "avg_pay_ratio": "Average payment-to-bill ratio",
    "limit_bal": "Credit limit",
    "age": "Age",
    "education": "Education",
    "bill_amt1": "Current statement balance",
    "bill_amt2": "Statement balance, 2 months prior",
    "bill_amt3": "Statement balance, 3 months prior",
    "bill_amt4": "Statement balance, 4 months prior",
    "bill_amt5": "Statement balance, 5 months prior",
    "bill_amt6": "Statement balance, 6 months prior",
    "pay_amt1": "Current payment amount",
    "pay_amt2": "Payment amount, 2 months prior",
    "pay_amt3": "Payment amount, 3 months prior",
    "pay_amt4": "Payment amount, 4 months prior",
    "pay_amt5": "Payment amount, 5 months prior",
    "pay_amt6": "Payment amount, 6 months prior",
}
