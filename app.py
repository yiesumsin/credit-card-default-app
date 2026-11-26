"""Credit risk assessment: a Streamlit front end for a Gradient Boosted Trees model trained in KNIME
and exported to PMML.

    credit_features.py   raw client columns -> the 16 model inputs (mirrors the KNIME workflow)
    pmml_model.py        loads the PMML file and returns P(default)
    metrics.py           confusion matrix, ROC curve and AUC
    ui.py                styling and Plotly charts
    model_meta.json      cross-validation results and the default cutoff
    sample_data/         synthetic clients for demonstrations

Streamlit reruns this script on every interaction; slow steps such as loading the model are cached.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from credit_features import (BILLS, EDUCATION_LABELS, MARRIAGE_LABELS, PAY_STATUS, PAY_STATUS_LABELS, PAYMENTS,
                             RAW_COLUMNS, add_features, find_label_column, missing_columns, risk_band,
                             standardise_columns)
from metrics import confusion_counts, roc_points, scores_from_counts
from pmml_model import PmmlModel
import ui

HERE = Path(__file__).parent
MODEL_PATH = HERE / "credit_card_default_classifier.pmml"

st.set_page_config(page_title="Credit risk assessment", layout="wide", initial_sidebar_state="collapsed")
ui.inject_css()


# ---------------------------------------------------------------------------------------------------------
# Model and saved results
# ---------------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model...")
def load_model(model_revision):
    """model_revision (the file's mtime) is only a cache key, so replacing the PMML reloads it."""
    return PmmlModel(MODEL_PATH)


def load_meta():
    return json.loads((HERE / "model_meta.json").read_text())


try:
    model_revision = MODEL_PATH.stat().st_mtime_ns
    model = load_model(model_revision)
except Exception as error:  # usually a missing Java runtime, which PyPMML needs
    st.error("The model could not be loaded. PyPMML needs Java 8 or newer; see Troubleshooting in the README.")
    st.exception(error)
    st.stop()

meta = load_meta()
BASE_RATE = meta["default_rate"]
DEFAULT_CUTOFF = float(meta["threshold"])
FINAL_RUN = "marriage_cv"


# ---------------------------------------------------------------------------------------------------------
# Session state: form values, example clients and the shared cutoff
# ---------------------------------------------------------------------------------------------------------
EXAMPLES = {
    "Low risk": dict(LIMIT_BAL=250_000, AGE=44, EDUCATION=2, MARRIAGE=1,
                     PAY_0=-1, PAY_2=-1, PAY_3=0, PAY_4=-1, PAY_5=-1, PAY_6=0,
                     BILL_AMT1=9_000, BILL_AMT2=8_500, BILL_AMT3=12_000, BILL_AMT4=7_000, BILL_AMT5=6_500, BILL_AMT6=8_000,
                     PAY_AMT1=8_500, PAY_AMT2=12_000, PAY_AMT3=7_000, PAY_AMT4=6_500, PAY_AMT5=8_000, PAY_AMT6=9_000),
    "Borderline": dict(LIMIT_BAL=120_000, AGE=33, EDUCATION=2, MARRIAGE=2,
                       PAY_0=1, PAY_2=0, PAY_3=0, PAY_4=1, PAY_5=0, PAY_6=0,
                       BILL_AMT1=58_000, BILL_AMT2=55_000, BILL_AMT3=52_000, BILL_AMT4=50_000, BILL_AMT5=47_000, BILL_AMT6=45_000,
                       PAY_AMT1=2_500, PAY_AMT2=2_000, PAY_AMT3=2_000, PAY_AMT4=0, PAY_AMT5=2_000, PAY_AMT6=2_000),
    "High risk": dict(LIMIT_BAL=60_000, AGE=29, EDUCATION=3, MARRIAGE=3,
                      PAY_0=2, PAY_2=2, PAY_3=2, PAY_4=3, PAY_5=2, PAY_6=0,
                      BILL_AMT1=61_000, BILL_AMT2=59_500, BILL_AMT3=58_000, BILL_AMT4=56_000, BILL_AMT5=53_000, BILL_AMT6=50_000,
                      PAY_AMT1=0, PAY_AMT2=0, PAY_AMT3=1_500, PAY_AMT4=0, PAY_AMT5=0, PAY_AMT6=1_000),
}

for key, value in EXAMPLES["Borderline"].items():
    st.session_state.setdefault(key, value)

# A new PMML file or a new default cutoff invalidates the stored result and resets both sliders.
assessment_revision = (model_revision, DEFAULT_CUTOFF)
if st.session_state.get("assessment_revision") != assessment_revision:
    st.session_state["cutoff"] = DEFAULT_CUTOFF
    st.session_state["batch_cutoff"] = DEFAULT_CUTOFF
    st.session_state.pop("assessment", None)
    st.session_state["assessment_revision"] = assessment_revision
st.session_state.setdefault("batch_cutoff", st.session_state["cutoff"])


def sync_cutoff(source, target):
    st.session_state[target] = st.session_state[source]


def load_example(name):
    st.session_state.update(EXAMPLES[name])
    st.session_state.pop("assessment", None)


MONTH_NAMES = ["Month 1 (latest)", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6 (oldest)"]

MODEL_INPUT_HELP = {
    "LIMIT_BAL": "Credit limit", "AGE": "Age", "PAY_0": "Latest repayment status",
    "avg_bill": "Mean of the six bills", "total_paid": "Sum of the six payments",
    "max_delay": "Worst status across six months", "utilisation": "Latest bill / credit limit",
    "pay_ratio": "Total paid / total billed", "n_late": "Months with a delay",
    "EDUCATION_1": "Graduate school (1/0)", "EDUCATION_2": "University (1/0)",
    "EDUCATION_3": "High school (1/0)", "EDUCATION_4": "Other (1/0)",
    "MARRIAGE_married": "Married (1/0)", "MARRIAGE_single": "Single (1/0)",
    "MARRIAGE_others": "Other marital status (1/0)",
}

CHART_CONFIG = {"displayModeBar": False}


def pct(value, digits=0):
    return f"{value:.{digits}%}"


# ---------------------------------------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------------------------------------
def drop_invalid_rows(data):
    """Coerce the raw columns to numbers and drop rows the model cannot score."""
    data[RAW_COLUMNS] = data[RAW_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid = (data[RAW_COLUMNS].isna().any(axis=1) | (data["LIMIT_BAL"] <= 0)
               | ~data["MARRIAGE"].isin(MARRIAGE_LABELS))
    if invalid.any():
        st.warning(f"Skipped {int(invalid.sum()):,} row(s) with missing or non-numeric values, a credit limit of "
                   "zero or less, or a MARRIAGE code other than 1, 2 or 3.")
    return data[~invalid].reset_index(drop=True)


def render_batch_results(data, cutoff):
    with st.spinner(f"Scoring {len(data):,} clients..."):
        probabilities = model.predict_default_probability(data)

    results = data.copy()
    results["default_probability"] = probabilities.round(4)
    results["predicted_default"] = (probabilities >= cutoff).astype(int)
    results["risk"] = [risk_band(p, cutoff) for p in probabilities]

    n_flagged = int(results["predicted_default"].sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(ui.card("Clients scored", f"{len(results):,}"), unsafe_allow_html=True)
    k2.markdown(ui.card("Flagged", f"{n_flagged:,}", f"{n_flagged / len(results):.1%} of clients"), unsafe_allow_html=True)
    k3.markdown(ui.card("Mean probability", pct(probabilities.mean(), 1), blue=True), unsafe_allow_html=True)
    k4.markdown(ui.card("Highest probability", pct(probabilities.max(), 1), blue=True), unsafe_allow_html=True)

    st.write("")
    chart1, chart2 = st.columns([1.5, 1])
    chart1.plotly_chart(ui.probability_histogram(probabilities, cutoff), width="stretch", config=CHART_CONFIG)
    chart2.plotly_chart(ui.risk_donut(n_flagged, len(results) - n_flagged), width="stretch", config=CHART_CONFIG)

    st.markdown("#### Results")
    table = results[results["risk"] == "High risk"] if st.checkbox("Show flagged clients only") else results
    lead = [c for c in ["ID"] if c in table.columns] + ["default_probability", "predicted_default", "risk"]
    table = table[lead + [c for c in table.columns if c not in lead]].sort_values("default_probability", ascending=False)
    st.dataframe(table, hide_index=True, width="stretch", height=360,
                 column_config={"default_probability": st.column_config.ProgressColumn(
                     "default_probability", min_value=0.0, max_value=1.0, format="%.3f")})
    st.download_button("Download results (CSV)", results.to_csv(index=False).encode("utf-8"),
                       file_name="credit_default_predictions.csv", mime="text/csv")

    label_name = find_label_column(data)
    if label_name is not None:
        render_file_evaluation(pd.to_numeric(data[label_name], errors="coerce"), probabilities, cutoff, label_name)


def render_file_evaluation(outcome, probabilities, cutoff, label_name):
    if not outcome.isin([0, 1]).all():
        st.warning(f"Column '{label_name}' must contain only 0 and 1, so the file was not evaluated.")
        return

    st.divider()
    st.markdown(f"### Performance on this file at a {pct(cutoff)} cutoff")
    tp, fp, tn, fn = confusion_counts(outcome, probabilities, cutoff)
    scores = scores_from_counts(tp, fp, tn, fn)
    fpr, tpr, auc = roc_points(outcome, probabilities)

    e1, e2, e3, e4 = st.columns(4)
    e1.markdown(ui.card("ROC AUC", f"{auc:.3f}", "0.5 = random ranking"), unsafe_allow_html=True)
    e2.markdown(ui.card("Recall", pct(scores["recall"], 1), "defaulters flagged", blue=True), unsafe_allow_html=True)
    e3.markdown(ui.card("Precision", pct(scores["precision"], 1), "flags that defaulted", blue=True), unsafe_allow_html=True)
    e4.markdown(ui.card("F1", f"{scores['f1']:.3f}", "harmonic mean of the two", blue=True), unsafe_allow_html=True)

    st.write("")
    m1, m2 = st.columns(2)
    m1.plotly_chart(ui.confusion_matrix_fig(tp, fp, tn, fn, "Confusion matrix"), width="stretch", config=CHART_CONFIG)
    m2.plotly_chart(ui.roc_fig([("This file", fpr, tpr, auc)], "ROC curve"), width="stretch", config=CHART_CONFIG)
    st.caption("Synthetic outcomes, or clients the model was trained on, make these figures optimistic. "
               "Cross-validated results are on the Model performance tab.")


# ---------------------------------------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------------------------------------
ui.hero("Credit risk assessment",
        "Estimated probability that a credit card client defaults on next month's payment.",
        ["Gradient Boosted Trees · KNIME", "Research prototype"])

tab_predict, tab_batch, tab_perf, tab_about = st.tabs(
    ["Client assessment", "Batch scoring", "Model performance", "About the model"])

# ---- Client assessment ----------------------------------------------------------------------------------
with tab_predict:
    st.markdown("### Client details")
    st.caption("Profile and six months of account history. Amounts are in New Taiwan dollars (NT$).")
    with st.expander("Load an example client"):
        st.caption("Replaces the form and clears the current result. Labels describe the profile, not the model's output.")
        for column, name in zip(st.columns(3), EXAMPLES):
            column.button(f"Load {name.lower()} example", key=f"example_{name}",
                          on_click=load_example, args=(name,), width="stretch")

    with st.form("client_assessment"):
        st.markdown("#### Profile")
        limit_col, age_col, education_col = st.columns([1.2, 0.8, 1])
        limit_col.number_input("Credit limit (NT$)", min_value=1, step=1000, key="LIMIT_BAL")
        age_col.number_input("Age", min_value=21, max_value=79, step=1, key="AGE")
        education_col.selectbox("Education", list(EDUCATION_LABELS), format_func=EDUCATION_LABELS.get, key="EDUCATION")
        st.selectbox("Marital status", list(MARRIAGE_LABELS), format_func=MARRIAGE_LABELS.get, key="MARRIAGE")

        st.markdown("#### Account history")
        st.caption("Latest month first. A negative bill is a credit balance.")
        for month, status, bill, payment in zip(MONTH_NAMES, PAY_STATUS, BILLS, PAYMENTS):
            st.markdown(f"**{month}**")
            status_col, bill_col, payment_col = st.columns([1.4, 1, 1])
            status_col.selectbox(f"Repayment status · {month}", list(PAY_STATUS_LABELS),
                                 format_func=PAY_STATUS_LABELS.get, key=status)
            bill_col.number_input(f"Bill (NT$) · {month}", step=500, key=bill)
            payment_col.number_input(f"Paid (NT$) · {month}", min_value=0, step=500, key=payment)

        submitted = st.form_submit_button("Calculate risk", type="primary", width="stretch")

    if submitted:
        raw_input = {column: st.session_state[column] for column in RAW_COLUMNS}
        with st.spinner("Scoring..."):
            probability = float(model.predict_default_probability(pd.DataFrame([raw_input]))[0])
        st.session_state["assessment"] = {"inputs": raw_input, "probability": probability}

    st.markdown("#### Decision cutoff")
    st.slider("Flag clients at or above this probability", 0.05, 0.80, step=0.01, key="cutoff", format="%.2f",
              on_change=sync_cutoff, args=("cutoff", "batch_cutoff"),
              help="Moves the decision line only; the predicted probability does not change.")
    st.caption(f"Default {pct(DEFAULT_CUTOFF)}. A lower cutoff flags more clients. Shared with Batch scoring.")
    cutoff = float(st.session_state["cutoff"])

    assessment = st.session_state.get("assessment")
    if assessment is None:
        st.info("Fill in the form and select Calculate risk.")
    else:
        st.divider()
        st.markdown("### Result")
        st.caption("Reflects the last submitted values. Recalculate after editing the form.")
        raw_one = pd.DataFrame([assessment["inputs"]])
        probability = assessment["probability"]
        st.markdown(ui.risk_card(probability >= cutoff, probability, cutoff), unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Default probability", pct(probability, 1))
        c2.metric("Training default rate", pct(BASE_RATE, 1))
        c3.metric("Decision cutoff", pct(cutoff))
        with st.expander("Submitted values"):
            st.dataframe(raw_one, hide_index=True, width="stretch")

        # Re-score the same client with each latest repayment status, holding everything else fixed.
        what_if_statuses = [-2, -1, 0, 1, 2, 3, 4]
        what_if_rows = pd.concat([raw_one.assign(PAY_0=s) for s in what_if_statuses], ignore_index=True)
        what_if_probs = model.predict_default_probability(what_if_rows)
        current = int(raw_one.iloc[0]["PAY_0"])
        current_index = what_if_statuses.index(current) if current in what_if_statuses else -1

        chart_col, table_col = st.columns([1.25, 1])
        with chart_col:
            st.plotly_chart(ui.what_if_chart([str(s) for s in what_if_statuses], what_if_probs, cutoff, current_index),
                            width="stretch", config=CHART_CONFIG)
            st.caption("Latest repayment status: −2 no consumption, −1 paid in full, 0 revolving, 1+ months late. "
                       "All other inputs are held at the submitted values. The darkest bar is the submitted status.")
        with table_col:
            st.markdown("#### Model inputs")
            features = add_features(raw_one).iloc[0]
            st.dataframe(pd.DataFrame({
                "Input": features.index,
                "Value": [f"{v:,.3f}".rstrip("0").rstrip(".") for v in features.values],
                "Meaning": [MODEL_INPUT_HELP[c] for c in features.index],
            }), hide_index=True, width="stretch", height=420)

# ---- Batch scoring --------------------------------------------------------------------------------------
with tab_batch:
    st.markdown("### Batch scoring")
    st.slider("Decision cutoff", 0.05, 0.80, step=0.01, key="batch_cutoff", format="%.2f",
              on_change=sync_cutoff, args=("batch_cutoff", "cutoff"),
              help="Flag clients at or above this probability. Shared with Client assessment.")
    st.caption(f"Default {pct(DEFAULT_CUTOFF)}. Shared with Client assessment.")
    st.markdown('<div class="note">One row per client. Required columns: '
                f'<code>{", ".join(RAW_COLUMNS)}</code>. Other columns, such as <code>ID</code>, are kept. '
                'Add a <code>default</code> column (1 = defaulted, 0 = did not) to evaluate the model on the file.</div>',
                unsafe_allow_html=True)
    st.write("")

    data = None
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
        except Exception as error:
            st.error(f"Could not read the file as CSV: {error}")

    if data is not None:
        data = standardise_columns(data)
        absent = missing_columns(data)
        if absent:
            st.error(f"Missing columns: {', '.join(absent)}")
        else:
            data = drop_invalid_rows(data)
            if data.empty:
                st.error("No valid rows to score.")
            else:
                render_batch_results(data, float(st.session_state["batch_cutoff"]))

# ---- Model performance ----------------------------------------------------------------------------------
with tab_perf:
    settings = meta["settings"]
    st.markdown("### Cross-validated performance")
    st.markdown(f'<div class="note">5-fold cross-validation in KNIME on {meta["trained_on_rows"]:,} clients: each '
                'client was scored by a model that had not seen it. The deployed model was then retrained on every row '
                f'with the same settings ({settings["trees"]} trees, depth {settings["tree_depth"]}, '
                f'learning rate {settings["learning_rate"]}).</div>', unsafe_allow_html=True)
    st.write("")

    runs = meta["runs"]
    choice = st.selectbox("Run", [key for key in runs if key != "no_smote_cv"], format_func=lambda k: runs[k]["label"])
    run = runs[choice]
    saved_cutoffs = [key for key in run if key.startswith("at_") and "tp" in run[key]]
    preferred = "at_0.30" if "at_0.30" in saved_cutoffs else saved_cutoffs[0]
    result_cutoff = st.selectbox("Cutoff", saved_cutoffs, index=saved_cutoffs.index(preferred),
                                 format_func=lambda key: pct(float(key[3:])))
    counts = run[result_cutoff]
    cutoff_label = f"at {pct(float(result_cutoff[3:]))}"
    st.caption("Stored results; the cutoff sliders on the other tabs do not affect them.")
    scores = scores_from_counts(**counts)

    p1, p2, p3, p4 = st.columns(4)
    p1.markdown(ui.card("ROC AUC", f"{run['auc']:.3f}", "all cutoffs"), unsafe_allow_html=True)
    p2.markdown(ui.card("Recall", pct(scores["recall"], 1), cutoff_label, blue=True), unsafe_allow_html=True)
    p3.markdown(ui.card("Precision", pct(scores["precision"], 1), cutoff_label, blue=True), unsafe_allow_html=True)
    p4.markdown(ui.card("F1", f"{scores['f1']:.3f}", cutoff_label, blue=True), unsafe_allow_html=True)
    st.write("")

    st.plotly_chart(ui.confusion_matrix_fig(counts["tp"], counts["fp"], counts["tn"], counts["fn"],
                                            f"Confusion matrix {cutoff_label}"),
                    width="stretch", config=CHART_CONFIG)
    st.caption("No ROC curve is stored for the final model; the AUC is the KNIME cross-validation value.")

    st.markdown("#### All runs")
    rows = []
    for r in runs.values():
        for cutoff_name in sorted(k for k in r if k.startswith("at_")):
            c = r[cutoff_name]
            s = scores_from_counts(**c) if "tp" in c else c
            rows.append({"Run": r["label"], "Cutoff": cutoff_name[3:], "ROC AUC": r["auc"],
                         "Recall": s["recall"], "Precision": s["precision"], "F1": s["f1"]})
    number = st.column_config.NumberColumn(format="%.3f")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={name: number for name in ("ROC AUC", "Recall", "Precision", "F1")})

    final = runs[FINAL_RUN].get(f"at_{DEFAULT_CUTOFF:.2f}")
    if final and "tp" in final:
        fs = scores_from_counts(**final)
        st.markdown(f'<div class="note"><b>Deployed configuration:</b> no SMOTE, cutoff {pct(DEFAULT_CUTOFF)}. '
                    f'Cross-validated recall {pct(fs["recall"], 1)}, precision {pct(fs["precision"], 1)}, '
                    f'F1 {fs["f1"]:.3f}.</div>', unsafe_allow_html=True)

# ---- About ----------------------------------------------------------------------------------------------
with tab_about:
    st.markdown("### About the model")
    st.markdown(f"""
The app takes a client's credit limit, age, education, marital status and six months of repayment status, bills and
payments, and returns the estimated probability of default next month. Clients at or above the cutoff
({pct(DEFAULT_CUTOFF)} by default) are flagged.

The model is a Gradient Boosted Trees classifier ({settings["trees"]} trees, depth {settings["tree_depth"]},
learning rate {settings["learning_rate"]}) trained in KNIME on the public *Default of Credit Card Clients* dataset
({meta["trained_on_rows"]:,} clients, {pct(BASE_RATE, 1)} defaulted) and exported as PMML.

#### Model inputs
The app derives the 16 inputs from the raw columns using the same formulas as the KNIME workflow.
""")
    st.dataframe(pd.DataFrame({
        "Input": ["avg_bill", "total_paid", "max_delay", "utilisation", "pay_ratio", "n_late", "PAY_0",
                  "EDUCATION_1 to 4", "MARRIAGE_married / single / others", "LIMIT_BAL, AGE"],
        "Definition": ["(BILL_AMT1 + … + BILL_AMT6) / 6", "PAY_AMT1 + … + PAY_AMT6",
                       "Largest of the six repayment statuses", "BILL_AMT1 / LIMIT_BAL",
                       "total_paid / total billed (divisor 1 when the total is 0)",
                       "Months with a status above 0", "Latest repayment status, unchanged",
                       "One-hot education; codes 0, 5 and 6 map to 4 (other)",
                       "One-hot marital status: 1 married, 2 single, 3 other", "Unchanged"],
    }), hide_index=True, width="stretch")
    st.markdown("""
#### Excluded columns
Sex is not used. The monthly bill, payment and status columns, apart from the latest status, are replaced by the
summary inputs above, which carry the same information with less redundancy.

#### Limitations
- The data comes from UCI dataset repository based off Credit Card Default Clients from Taiwan in 2005. This is a proof of concept, not suitable for credit decisions.
- A cross-validated ROC AUC of about 0.78 means the model ranks risk usefully but imperfectly. At the 0.30 cutoff,
  roughly half of the flagged clients do not default.
- Probabilities are only meaningful for clients similar to the training population.
- The bundled 1,000-client sample is synthetic, with outcomes drawn from the model's own probabilities. Scores on it
  demonstrate the app, not the model's accuracy.
""")
