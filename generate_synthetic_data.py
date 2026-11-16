"""Creates sample_data/synthetic_clients_1000.csv: 1,000 made-up credit card clients for testing the app.

Nobody in this file is real. The rows are generated from simple rules that imitate the shape of the public UCI
"Default of Credit Card Clients" data (credit limits, repayment statuses that persist from month to month, bills
that move with the limit, payments that depend on how well the client is repaying).

HOW THE 'default' LABEL IS MADE (important for your presentation)
The label is drawn at random using the MODEL's own predicted probability (a client with a 40% predicted risk
becomes a defaulter 40% of the time). That makes the file useful for demonstrating the app - upload it and
the performance section works - but the scores on it are NOT evidence that the model is good. The real evidence
is the 5-fold cross-validation on the real data (see the Model performance tab).

Run it again with:   python generate_synthetic_data.py
(needs Java installed, because the label step uses the PMML model)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from credit_features import BILLS, PAY_STATUS, PAYMENTS, RAW_COLUMNS
from pmml_model import PmmlModel

HERE = Path(__file__).parent
N_ROWS = 1000
SEED = 2026
rng = np.random.default_rng(SEED)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---- 1. Who the client is -------------------------------------------------------------------------------
limit_bal = np.clip(np.round(rng.lognormal(np.log(110_000), 0.85, N_ROWS), -4), 10_000, 800_000)
age = np.clip(np.round(rng.normal(35, 9, N_ROWS) + rng.gamma(1.2, 1.5, N_ROWS) - 1.8), 21, 75)
education = rng.choice([1, 2, 3, 4], size=N_ROWS, p=[0.35, 0.47, 0.16, 0.02])

# ---- 2. A hidden "financial stress" level that drives behaviour (not saved in the file) -------------------
stress = rng.normal(0, 1, N_ROWS)

# ---- 3. Repayment status for 6 months. Status persists: this month depends on last month plus noise -------
# Month index 5 is the oldest (PAY_6), 0 is the newest (PAY_0).
latent = np.zeros((N_ROWS, 6))
previous = 0.55 * stress + rng.normal(0, 0.8, N_ROWS)
for month in range(5, -1, -1):
    previous = 0.62 * previous + 0.55 * stress + rng.normal(0, 0.62, N_ROWS)
    latent[:, month] = previous


def latent_to_status(column):
    """Turn a continuous score into the -2 ... 8 status codes, using shares close to the real data."""
    shares = [(-2, 0.092), (-1, 0.19), (0, 0.491), (1, 0.123), (2, 0.089), (3, 0.011), (4, 0.003),
              (5, 0.001), (6, 0.0005), (7, 0.0003), (8, 0.0002)]
    cut = np.cumsum([s for _, s in shares])
    ranks = column.argsort().argsort() / (len(column) - 1)
    status = np.full(len(column), shares[-1][0], dtype=int)       # anything left over gets the top status
    lower = 0.0
    for (code, _), upper in zip(shares, cut):
        status[(ranks >= lower) & (ranks <= upper)] = code
        lower = upper
    return status


status = np.column_stack([latent_to_status(latent[:, m]) for m in range(6)])      # columns: PAY_0 ... PAY_6

# ---- 4. Bills: a share of the limit that grows a bit with stress, with month-to-month wobble --------------
usage = np.clip(rng.beta(1.6, 3.0, N_ROWS) * (0.7 + 0.35 * sigmoid(stress)), 0, 1.1)
bills = np.zeros((N_ROWS, 6))
level = limit_bal * usage
for month in range(5, -1, -1):                        # oldest bill first
    level = level * rng.normal(1.01, 0.12, N_ROWS)
    bills[:, month] = level
bills[status == -2] = rng.uniform(0, 400, size=(status == -2).sum())        # "no consumption" -> almost no bill
credit_balance = rng.random(bills.shape) < 0.02                              # a few credit balances (negative bills)
bills[credit_balance] = -rng.uniform(50, 6000, credit_balance.sum())
bills = np.clip(np.round(bills), -60_000, 1.3 * limit_bal[:, None])

# ---- 5. Payments: PAY_AMT_t is paid against the previous month's bill (BILL_AMT_(t+1)) --------------------
payments = np.zeros((N_ROWS, 6))
for month in range(6):
    previous_bill = np.clip(bills[:, min(month + 1, 5)], 0, None)
    code = status[:, month]
    share = np.where(code <= -1, rng.uniform(0.6, 1.0, N_ROWS),                     # paid in full
            np.where(code == 0, rng.beta(1.2, 9, N_ROWS) + 0.02,                   # minimum-ish payment
                     np.where(rng.random(N_ROWS) < 0.45, 0.0, rng.beta(1.0, 12, N_ROWS))))  # late: often nothing
    payments[:, month] = np.round(previous_bill * share)

# ---- 6. Put it in the same column layout as the real file -------------------------------------------------
data = pd.DataFrame({"ID": np.arange(1, N_ROWS + 1), "LIMIT_BAL": limit_bal.astype(int), "EDUCATION": education,
                     "AGE": age.astype(int)})
for i, name in enumerate(PAY_STATUS):
    data[name] = status[:, i]
for i, name in enumerate(BILLS):
    data[name] = bills[:, i].astype(int)
for i, name in enumerate(PAYMENTS):
    data[name] = payments[:, i].astype(int)
# Synthetic marital statuses, independent of repayment stress; no causal assumption.
data["MARRIAGE"] = rng.choice([1, 2, 3], N_ROWS)
data = data[["ID"] + RAW_COLUMNS]

# ---- 7. The label: drawn from the model's own probability (see the note at the top of this file) ----------
model = PmmlModel(HERE / "credit_card_default_classifier.pmml")
probability = model.predict_default_probability(data)
with_label = data.copy()
with_label["default"] = (rng.random(N_ROWS) < probability).astype(int)

(HERE / "sample_data").mkdir(exist_ok=True)
with_label.to_csv(HERE / "sample_data" / "synthetic_clients_1000.csv", index=False)
data.to_csv(HERE / "sample_data" / "synthetic_clients_1000_no_labels.csv", index=False)
print(f"Wrote {N_ROWS} rows. Default rate {with_label['default'].mean():.1%}, mean model probability {probability.mean():.1%}")
