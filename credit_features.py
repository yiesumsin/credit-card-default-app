"""Feature engineering for the credit card default app.

The PMML file exported from KNIME contains ONLY the trained trees. It expects 16 model inputs:

    LIMIT_BAL, AGE, PAY_0, avg_bill, total_paid, max_delay, utilisation, pay_ratio, n_late,
    EDUCATION_1, EDUCATION_2, EDUCATION_3, EDUCATION_4,
    MARRIAGE_married, MARRIAGE_single, MARRIAGE_others

Everything except LIMIT_BAL, AGE and PAY_0 is calculated from the raw client columns. This file repeats the
KNIME Math Formula / Column Aggregator / One to Many steps exactly, so the app prepares a client the same way
the model was trained. If you change a formula in KNIME, change it here too.
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------------------
# Column names used by the dataset (UCI "Default of Credit Card Clients").
# PAY_0 is the repayment status of the most recent month; PAY_2 ... PAY_6 go back in time (there is no PAY_1).
# ---------------------------------------------------------------------------------------------------------
PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]       # bill statement amount, month 1 (latest) to 6
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]     # amount paid, month 1 (latest) to 6

# The raw columns a client file must contain. (SEX is not used. MARRIAGE must be 1, 2 or 3.)
RAW_COLUMNS = ["LIMIT_BAL", "AGE", "EDUCATION", "MARRIAGE"] + PAY_STATUS + BILLS + PAYMENTS

# The 16 columns the PMML model reads (the order does not matter to PMML, it matches by name).
MODEL_COLUMNS = [
    "LIMIT_BAL", "AGE", "PAY_0", "avg_bill", "total_paid", "max_delay", "utilisation", "pay_ratio", "n_late",
    "EDUCATION_1", "EDUCATION_2", "EDUCATION_3", "EDUCATION_4",
    "MARRIAGE_married", "MARRIAGE_single", "MARRIAGE_others",
]

# Names of the optional label column in a test file (any of these is accepted).
LABEL_COLUMNS = ["default", "default payment next month", "DEFAULT"]

MARRIAGE_LABELS = {1: "Married", 2: "Single", 3: "Others"}
EDUCATION_LABELS = {1: "Graduate school", 2: "University", 3: "High school", 4: "Others / unknown"}
PAY_STATUS_LABELS = {
    -2: "-2  No consumption",
    -1: "-1  Paid in full",
    0: "0  Revolving credit (paid the minimum)",
    1: "1  Payment delayed 1 month",
    2: "2  Payment delayed 2 months",
    3: "3  Payment delayed 3 months",
    4: "4  Payment delayed 4 months",
    5: "5  Payment delayed 5 months",
    6: "6  Payment delayed 6 months",
    7: "7  Payment delayed 7 months",
    8: "8  Payment delayed 8+ months",
}


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept slightly different column spellings, e.g. PAY_1 (some copies of the file) instead of PAY_0."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "PAY_0" not in df.columns and "PAY_1" in df.columns:
        df = df.rename(columns={"PAY_1": "PAY_0"})
    return df


def missing_columns(df: pd.DataFrame) -> list:
    """Raw columns that a file needs but does not have (empty list = the file is fine)."""
    return [c for c in RAW_COLUMNS if c not in df.columns]


def find_label_column(df: pd.DataFrame):
    """Return the name of the true-outcome column if the file has one, otherwise None."""
    for name in LABEL_COLUMNS:
        if name in df.columns:
            return name
    return None


def add_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw client columns in, the 16 model inputs out (same formulas as the KNIME workflow)."""
    d = raw[RAW_COLUMNS].astype(float).copy()
    out = pd.DataFrame(index=d.index)

    out["LIMIT_BAL"] = d["LIMIT_BAL"]
    out["AGE"] = d["AGE"]
    out["PAY_0"] = d["PAY_0"]

    # avg_bill = (BILL_AMT1 + ... + BILL_AMT6) / 6
    out["avg_bill"] = d[BILLS].sum(axis=1) / 6
    # total_paid = PAY_AMT1 + ... + PAY_AMT6
    out["total_paid"] = d[PAYMENTS].sum(axis=1)
    # max_delay = the worst (largest) repayment status of the six months (KNIME Column Aggregator, maximum)
    out["max_delay"] = d[PAY_STATUS].max(axis=1)
    # utilisation = latest bill as a share of the credit limit
    out["utilisation"] = d["BILL_AMT1"] / d["LIMIT_BAL"]
    # pay_ratio = total_paid / bill total, using 1 as the divisor when the bill total is exactly 0.
    # (Negative bill totals - credit balances - are left as they are, as in the KNIME formula.)
    bill_total = d[BILLS].sum(axis=1)
    out["pay_ratio"] = out["total_paid"] / bill_total.where(bill_total != 0, 1.0)
    # n_late = how many of the six months had a payment delay (status greater than 0)
    out["n_late"] = (d[PAY_STATUS] > 0).sum(axis=1).astype(float)

    # EDUCATION: the undocumented codes 0, 5 and 6 are recoded to 4 ("others"), then one-hot encoded
    education = d["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    for code in (1, 2, 3, 4):
        out[f"EDUCATION_{code}"] = (education == code).astype(float)

    # Do not silently assign undocumented codes to a marital-status category.
    if not d["MARRIAGE"].isin(MARRIAGE_LABELS).all():
        raise ValueError("MARRIAGE must be 1 (married), 2 (single), or 3 (others).")
    for code, label in ((1, "married"), (2, "single"), (3, "others")):
        out[f"MARRIAGE_{label}"] = (d["MARRIAGE"] == code).astype(float)

    return out[MODEL_COLUMNS]


def risk_band(probability: float, threshold: float) -> str:
    """Plain-language label used by the app: 'High risk' at or above the threshold, otherwise 'Low risk'."""
    return "High risk" if probability >= threshold else "Low risk"
