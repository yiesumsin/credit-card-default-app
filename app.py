"""Credit Card Default Predictor: a Streamlit app that scores clients with the Gradient Boosted Trees model
trained in KNIME and exported as a PMML file.

HOW A STREAMLIT APP WORKS (read this first if you are new)
Streamlit runs this file from top to bottom every time someone moves a slider or clicks a button. Anything that
is slow to build (like loading the model) is wrapped in @st.cache_resource so it only happens once.

FILE MAP
    app.py                  this file: the page layout and what happens on each tab
    credit_features.py      turns raw client columns into the 13 model inputs (same formulas as KNIME)
    pmml_model.py           loads the PMML file and returns a probability of default
    ui.py                   colours, custom CSS and the Plotly charts
    metrics.py              confusion matrix / ROC / AUC calculations
    model_meta.json         the cross-validation results from KNIME and the chosen cutoff (0.25)
    reference_roc.csv       points for the reference ROC curve
    sample_data/            1,000 synthetic clients for testing and presenting
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from credit_features import (EDUCATION_LABELS, LABEL_COLUMNS, MODEL_COLUMNS, PAY_STATUS, PAY_STATUS_LABELS, BILLS,
                             PAYMENTS, RAW_COLUMNS, add_features, find_label_column, missing_columns,
                             standardise_columns)
from metrics import confusion_counts, roc_points, scores_from_counts
from pmml_model import PmmlModel
import ui

HERE = Path(__file__).parent

st.set_page_config(page_title="Credit Card Default Predictor", page_icon="💳", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()


# =========================================================================================================
# 1. Load the model and the saved results (cached, so this happens once per session)
# =========================================================================================================
@st.cache_resource(show_spinner="Loading the model...")
def load_model():
    return PmmlModel(HERE / "credit_card_default_classifier.pmml")


@st.cache_data
def load_meta():
    return json.loads((HERE / "model_meta.json").read_text())


@st.cache_data
def load_reference_roc():
    return pd.read_csv(HERE / "reference_roc.csv")


try:
    model = load_model()
except Exception as error:      # most often: Java is not installed (PyPMML needs it)
    st.error("The model could not be loaded. PyPMML needs Java 8 or newer. See the README, section 'Troubleshooting'.")
    st.exception(error)
    st.stop()

meta = load_meta()
BASE_RATE = meta["default_rate"]

# =========================================================================================================
# 2. Sidebar: the decision cutoff and the client inputs
#    Every widget has a key, and st.session_state[key] holds its value. That lets the 'example client'
#    buttons fill the whole form at once.
# =========================================================================================================
EXAMPLES = {
    "Low risk": dict(LIMIT_BAL=250_000, AGE=44, EDUCATION=2,
                     PAY_0=-1, PAY_2=-1, PAY_3=0, PAY_4=-1, PAY_5=-1, PAY_6=0,
                     BILL_AMT1=9_000, BILL_AMT2=8_500, BILL_AMT3=12_000, BILL_AMT4=7_000, BILL_AMT5=6_500, BILL_AMT6=8_000,
                     PAY_AMT1=8_500, PAY_AMT2=12_000, PAY_AMT3=7_000, PAY_AMT4=6_500, PAY_AMT5=8_000, PAY_AMT6=9_000),
    "Borderline": dict(LIMIT_BAL=120_000, AGE=33, EDUCATION=2,
                     PAY_0=1, PAY_2=0, PAY_3=0, PAY_4=1, PAY_5=0, PAY_6=0,
                     BILL_AMT1=58_000, BILL_AMT2=55_000, BILL_AMT3=52_000, BILL_AMT4=50_000, BILL_AMT5=47_000, BILL_AMT6=45_000,
                     PAY_AMT1=2_500, PAY_AMT2=2_000, PAY_AMT3=2_000, PAY_AMT4=0, PAY_AMT5=2_000, PAY_AMT6=2_000),
    "High risk": dict(LIMIT_BAL=60_000, AGE=29, EDUCATION=3,
                      PAY_0=2, PAY_2=2, PAY_3=2, PAY_4=3, PAY_5=2, PAY_6=0,
                      BILL_AMT1=61_000, BILL_AMT2=59_500, BILL_AMT3=58_000, BILL_AMT4=56_000, BILL_AMT5=53_000, BILL_AMT6=50_000,
                      PAY_AMT1=0, PAY_AMT2=0, PAY_AMT3=1_500, PAY_AMT4=0, PAY_AMT5=0, PAY_AMT6=1_000),
}

# Start with the Borderline client the first time the app opens
for key, value in EXAMPLES["Borderline"].items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("cutoff", float(meta["threshold"]))


def load_example(name):
    """Button callback: copy an example client into the widgets."""
    for key, value in EXAMPLES[name].items():
        st.session_state[key] = value


MONTH_NAMES = ["Month 1 (latest)", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6 (oldest)"]

with st.sidebar:
    st.markdown("## 👤 Client details")
    st.caption("Change any value and the prediction updates straight away.")

    st.markdown("**Quick examples**")
    for name in EXAMPLES:
        st.button(f"{name} client", key=f"example_{name}", on_click=load_example, args=(name,), width="stretch")

    with st.expander("🧾 Profile", expanded=True):
        st.slider("Credit limit (NT$)", 10_000, 1_000_000, step=10_000, key="LIMIT_BAL",
                  help="The credit limit granted to the client (the dataset uses New Taiwan dollars).")
        st.slider("Age", 21, 79, key="AGE")
        st.selectbox("Education", list(EDUCATION_LABELS), format_func=EDUCATION_LABELS.get, key="EDUCATION")

    with st.expander("⏱️ Repayment status, last 6 months", expanded=True):
        st.caption("-2 no consumption, -1 paid in full, 0 paid the minimum, 1 or more = months of delay.")
        for name, month in zip(PAY_STATUS, MONTH_NAMES):
            st.selectbox(month, list(PAY_STATUS_LABELS), format_func=PAY_STATUS_LABELS.get, key=name)

    with st.expander("💳 Bill statement amounts (NT$)", expanded=False):
        for i, name in enumerate(BILLS):
            st.slider(MONTH_NAMES[i], -50_000, 700_000, step=500, key=name,
                      help="A negative amount is a credit balance (the client overpaid).")

    with st.expander("💵 Amounts paid (NT$)", expanded=False):
        for i, name in enumerate(PAYMENTS):
            st.slider(MONTH_NAMES[i], 0, 300_000, step=500, key=name)

    with st.expander("⚙️ Decision cutoff", expanded=False):
        st.slider("Flag as defaulter at or above", 0.05, 0.80, step=0.01, key="cutoff",
                  help="The final model was evaluated at 0.25. A lower cutoff catches more defaulters "
                       "but raises more false alarms.")
        st.caption(f"Chosen cutoff from the cross-validation: **{meta['threshold']:.2f}**")

CUTOFF = float(st.session_state["cutoff"])

# =========================================================================================================
# 3. Page header
# =========================================================================================================
ui.hero("Credit Card Default Predictor",
        "Estimates the chance that a credit card client misses a payment next month.",
        ["Gradient Boosted Trees", "Trained in KNIME", f"Cutoff {CUTOFF:.0%}", "Proof of concept"])

tab_predict, tab_batch, tab_perf, tab_about = st.tabs(
    ["🔮 Predict one client", "📂 Score a CSV file", "📊 Model performance", "ℹ️ About"])

MODEL_INPUT_HELP = {
    "LIMIT_BAL": "Credit limit", "AGE": "Age", "PAY_0": "Latest repayment status",
    "avg_bill": "Average of six bills", "total_paid": "Sum of the six payments",
    "max_delay": "Worst status of the six months", "utilisation": "Latest bill / credit limit",
    "pay_ratio": "Total paid / total billed", "n_late": "Months with a delay",
    "EDUCATION_1": "Graduate school (1/0)", "EDUCATION_2": "University (1/0)",
    "EDUCATION_3": "High school (1/0)", "EDUCATION_4": "Others (1/0)",
}

# =========================================================================================================
# 4. TAB 1: predict one client
# =========================================================================================================
with tab_predict:
    raw_one = pd.DataFrame([{column: st.session_state[column] for column in RAW_COLUMNS}])
    probability = float(model.predict_default_probability(raw_one)[0])
    is_high = probability >= CUTOFF

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown(ui.risk_card(is_high, probability, CUTOFF), unsafe_allow_html=True)
        st.write("")
        c1, c2, c3 = st.columns(3)
        c1.markdown(ui.card("Default probability", f"{probability:.1%}", "chance of missing a payment"), unsafe_allow_html=True)
        c2.markdown(ui.card("Compared with average", f"{probability / BASE_RATE:.1f}×",
                            f"average client: {BASE_RATE:.1%}", blue=True), unsafe_allow_html=True)
        c3.markdown(ui.card("Decision cutoff", f"{CUTOFF:.0%}", "flag at or above this", blue=True), unsafe_allow_html=True)
    with right:
        st.plotly_chart(ui.gauge(probability, CUTOFF), width="stretch", config={"displayModeBar": False})

    st.write("")
    # --- What-if chart: re-score the same client with different latest repayment statuses -------------------
    what_if_statuses = [-2, -1, 0, 1, 2, 3, 4]
    what_if_rows = pd.concat([raw_one.assign(PAY_0=s) for s in what_if_statuses], ignore_index=True)
    what_if_probs = model.predict_default_probability(what_if_rows)
    current = st.session_state["PAY_0"]
    chart_col, table_col = st.columns([1.25, 1])
    with chart_col:
        st.plotly_chart(ui.what_if_chart([str(s) for s in what_if_statuses],
                                         what_if_probs, CUTOFF, what_if_statuses.index(current) if current in what_if_statuses else -1),
                        width="stretch", config={"displayModeBar": False})
        st.caption("Horizontal axis: latest repayment status code (-2 no consumption, -1 paid in full, 0 minimum, 1 or more = months late). "
                   "Only this status changes; everything else stays as in the sidebar. The darkest bar is the client's current status.")
    with table_col:
        st.markdown("#### What the model sees")
        features = add_features(raw_one).iloc[0]
        view = pd.DataFrame({"Model input": features.index,
                             "Value": [f"{v:,.3f}".rstrip("0").rstrip(".") for v in features.values],
                             "Meaning": [MODEL_INPUT_HELP[c] for c in features.index]})
        st.dataframe(view, hide_index=True, width="stretch", height=420)

# =========================================================================================================
# 5. TAB 2: score a whole CSV file
# =========================================================================================================
with tab_batch:
    st.markdown("### Score many clients at once")
    st.markdown('<div class="note">Upload a CSV with one row per client. Required columns: '
                f'<code>{", ".join(RAW_COLUMNS)}</code>. Extra columns (like <code>ID</code>) are kept. '
                'If the file also has a <code>default</code> column (1 = defaulted, 0 = did not) the app shows '
                'how well the model did on it.</div>', unsafe_allow_html=True)
    st.write("")
    source = st.radio("Where should the data come from?",
                      ["Upload my own CSV", "Use the sample file with outcomes (1,000 synthetic clients)",
                       "Use the sample file without outcomes (1,000 synthetic clients)"], horizontal=False)

    data = None
    if source == "Upload my own CSV":
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded is not None:
            try:
                data = pd.read_csv(uploaded)
            except Exception as error:
                st.error(f"That file could not be read as a CSV: {error}")
    elif "without" in source:
        data = pd.read_csv(HERE / "synthetic_clients_1000_no_labels.csv")
    else:
        data = pd.read_csv(HERE / "synthetic_clients_1000.csv")

    if data is not None:
        data = standardise_columns(data)
        absent = missing_columns(data)
        if absent:
            st.error(f"The file is missing these columns: {', '.join(absent)}")
        else:
            # --- clean: numbers only, positive credit limit ------------------------------------------------
            data[RAW_COLUMNS] = data[RAW_COLUMNS].apply(pd.to_numeric, errors="coerce")
            bad = data[RAW_COLUMNS].isna().any(axis=1) | (data["LIMIT_BAL"] <= 0)
            if bad.any():
                st.warning(f"{int(bad.sum()):,} row(s) with missing, non-numeric or zero credit-limit values were left out.")
                data = data[~bad].reset_index(drop=True)
            if len(data) == 0:
                st.error("No usable rows were left to score.")
                st.stop()

            with st.spinner(f"Scoring {len(data):,} clients..."):
                probabilities = model.predict_default_probability(data)
            results = data.copy()
            results["default_probability"] = probabilities.round(4)
            results["predicted_default"] = (probabilities >= CUTOFF).astype(int)
            results["risk"] = ["High risk" if p >= CUTOFF else "Low risk" for p in probabilities]

            n_high = int(results["predicted_default"].sum())
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(ui.card("Clients scored", f"{len(results):,}", "rows used"), unsafe_allow_html=True)
            k2.markdown(ui.card("Flagged as defaulters", f"{n_high:,}", f"{n_high / len(results):.1%} of clients"), unsafe_allow_html=True)
            k3.markdown(ui.card("Average probability", f"{probabilities.mean():.1%}", "mean risk", blue=True), unsafe_allow_html=True)
            k4.markdown(ui.card("Highest risk", f"{probabilities.max():.1%}", "one client", blue=True), unsafe_allow_html=True)

            st.write("")
            chart1, chart2 = st.columns([1.5, 1])
            chart1.plotly_chart(ui.probability_histogram(probabilities, CUTOFF), width="stretch", config={"displayModeBar": False})
            chart2.plotly_chart(ui.risk_donut(n_high, len(results) - n_high), width="stretch", config={"displayModeBar": False})

            # --- results table and download ------------------------------------------------------------------
            st.markdown("#### Results")
            show_only_high = st.checkbox("Show only high-risk clients", value=False)
            table = results[results["risk"] == "High risk"] if show_only_high else results
            first_columns = [c for c in ["ID"] if c in table.columns] + ["default_probability", "predicted_default", "risk"]
            table = table[first_columns + [c for c in table.columns if c not in first_columns]]
            table = table.sort_values("default_probability", ascending=False)
            st.dataframe(table, hide_index=True, width="stretch", height=360,
                         column_config={"default_probability": st.column_config.ProgressColumn(
                             "default_probability", min_value=0.0, max_value=1.0, format="%.3f")})
            st.download_button("⬇️ Download all results (CSV)", results.to_csv(index=False).encode("utf-8"),
                               file_name="credit_default_predictions.csv", mime="text/csv")

            # --- how well did the model do on this file? (only when the true outcome is in the file) ------------
            label_name = find_label_column(data)
            if label_name is not None:
                outcome = pd.to_numeric(data[label_name], errors="coerce")
                if outcome.isin([0, 1]).all():
                    st.markdown("---")
                    st.markdown(f"### How the model did on this file (cutoff {CUTOFF:.0%})")
                    tp, fp, tn, fn = confusion_counts(outcome, probabilities, CUTOFF)
                    sc = scores_from_counts(tp, fp, tn, fn)
                    fpr, tpr, auc = roc_points(outcome, probabilities)
                    e1, e2, e3, e4 = st.columns(4)
                    e1.markdown(ui.card("ROC AUC", f"{auc:.3f}", "ranking quality (0.5 = coin flip)"), unsafe_allow_html=True)
                    e2.markdown(ui.card("Recall", f"{sc['recall']:.1%}", "defaulters caught", blue=True), unsafe_allow_html=True)
                    e3.markdown(ui.card("Precision", f"{sc['precision']:.1%}", "flags that were right", blue=True), unsafe_allow_html=True)
                    e4.markdown(ui.card("F1 score", f"{sc['f1']:.3f}", "balance of the two", blue=True), unsafe_allow_html=True)
                    st.write("")
                    m1, m2 = st.columns(2)
                    m1.plotly_chart(ui.confusion_matrix_fig(tp, fp, tn, fn, "Confusion matrix on this file"),
                                    width="stretch", config={"displayModeBar": False})
                    m2.plotly_chart(ui.roc_fig([("This file", fpr, tpr, auc)], "ROC curve on this file"),
                                    width="stretch", config={"displayModeBar": False})
                    st.caption("If this file contains clients the model was trained on, or synthetic outcomes, these numbers "
                               "look better than they would on new clients. For the honest estimate see the Model performance tab.")
                else:
                    st.warning(f"The '{label_name}' column should contain only 0 and 1, so the evaluation was skipped.")

# =========================================================================================================
# 6. TAB 3: model performance (results saved from the KNIME cross-validation)
# =========================================================================================================
with tab_perf:
    st.markdown("### How good is the model?")
    st.markdown('<div class="note">These results come from <b>5-fold cross-validation in KNIME</b> on the real dataset '
                '(30,000 clients): every client was scored by a model that had not seen them. The deployed model was then '
                'retrained on all 30,000 rows, with the same settings: 100 trees, depth 3, learning rate 0.03.</div>',
                unsafe_allow_html=True)
    st.write("")

    runs = meta["runs"]
    choice = st.selectbox("Which result do you want to see?", list(runs), format_func=lambda k: runs[k]["label"])
    run = runs[choice]
    counts = run["at_0.25"]
    sc = scores_from_counts(**counts)

    p1, p2, p3, p4 = st.columns(4)
    p1.markdown(ui.card("ROC AUC", f"{run['auc']:.3f}", "ranking quality"), unsafe_allow_html=True)
    p2.markdown(ui.card("Recall", f"{sc['recall']:.1%}", "at the 0.25 cutoff", blue=True), unsafe_allow_html=True)
    p3.markdown(ui.card("Precision", f"{sc['precision']:.1%}", "at the 0.25 cutoff", blue=True), unsafe_allow_html=True)
    p4.markdown(ui.card("F1 score", f"{sc['f1']:.3f}", "at the 0.25 cutoff", blue=True), unsafe_allow_html=True)
    st.write("")

    roc = load_reference_roc()
    a, b = st.columns(2)
    a.plotly_chart(ui.confusion_matrix_fig(counts["tp"], counts["fp"], counts["tn"], counts["fn"],
                                           f"Confusion matrix at the 0.25 cutoff"),
                   width="stretch", config={"displayModeBar": False})
    b.plotly_chart(ui.roc_fig([("Final model, cross-validated", roc["fpr"], roc["tpr"], meta["replica_roc_auc"])],
                              "ROC curve (5-fold cross-validated)"),
                   width="stretch", config={"displayModeBar": False})
    st.caption(f"The confusion matrices and the AUC in the cards are the KNIME results. The ROC curve was redrawn in Python "
               f"with the same settings and folds (AUC {meta['replica_roc_auc']:.3f}, KNIME reported "
               f"{runs['no_smote_cv']['auc']:.3f}), because KNIME does not export the curve points.")

    # --- comparison table of all runs --------------------------------------------------------------------------
    st.markdown("#### All runs side by side")
    rows = []
    for key, r in runs.items():
        for cutoff_name in ("at_0.25", "at_0.50"):
            c = r[cutoff_name]
            s = scores_from_counts(**c) if "tp" in c else c
            rows.append({"Run": r["label"], "Cutoff": cutoff_name[3:], "ROC AUC": r["auc"], "Recall": s["recall"],
                         "Precision": s["precision"], "F1": s["f1"]})
    comparison = pd.DataFrame(rows)
    st.dataframe(comparison, hide_index=True, width="stretch",
                 column_config={"ROC AUC": st.column_config.NumberColumn(format="%.3f"),
                                "Recall": st.column_config.NumberColumn(format="%.3f"),
                                "Precision": st.column_config.NumberColumn(format="%.3f"),
                                "F1": st.column_config.NumberColumn(format="%.3f")})
    st.markdown('<div class="note"><b>Reading the table.</b> SMOTE (creating synthetic defaulters in the training data) did '
                'not improve the ranking (AUC 0.773 against 0.779) and made the probabilities too high, so the final model was '
                'trained without it and the cutoff was lowered to 0.25 instead. At 0.25 the model catches about 63% of '
                'defaulters, and about 47% of the clients it flags really default.</div>', unsafe_allow_html=True)

# =========================================================================================================
# 7. TAB 4: about
# =========================================================================================================
with tab_about:
    st.markdown("### About this app")
    st.markdown(f"""
**What it does.** It takes a client's credit limit, age, education, six months of repayment status, bills and payments, and
returns the estimated probability that the client will default next month. A client at or above the cutoff
(**{meta['threshold']:.0%}** by default) is flagged as a likely defaulter.

**The model.** A Gradient Boosted Trees model with 100 trees, depth 3 and learning rate 0.03, trained in KNIME on the
public *Default of Credit Card Clients* dataset (30,000 clients, 22.1% defaulters) and exported as a PMML file.

**The 13 model inputs.** The app calculates these from the raw columns, with the same formulas as the KNIME workflow:
""")
    formulas = pd.DataFrame({
        "Input": ["avg_bill", "total_paid", "max_delay", "utilisation", "pay_ratio", "n_late", "PAY_0", "EDUCATION_1 to 4",
                  "LIMIT_BAL, AGE"],
        "How it is made": ["(BILL_AMT1 + ... + BILL_AMT6) / 6", "PAY_AMT1 + ... + PAY_AMT6",
                           "Largest of the six repayment statuses", "BILL_AMT1 / LIMIT_BAL",
                           "total_paid / total billed (1 if the bill total is 0)",
                           "Count of months with a status above 0", "Latest repayment status, used as it is",
                           "One-hot encoding of education (codes 0, 5, 6 are treated as 'others')", "Used as they are"]})
    st.dataframe(formulas, hide_index=True, width="stretch")
    st.markdown("""
**What was left out, and why.** Sex and marital status are not used, so clients are not scored on personal characteristics.
The individual monthly bill, payment and status columns (except the latest status) were dropped because they are
highly correlated with the summary columns above, so keeping both would only repeat the same information.

**Limits you should state when presenting.**
- The data is from Taiwan in 2005. It is a proof of concept of the method, **not EPF data**, and not a tool for real credit decisions.
- Cross-validated ROC AUC is about 0.78: the model ranks risk usefully but is far from perfect. At the 0.25 cutoff roughly half of
  the flagged clients do not default.
- The probability is only meaningful for clients similar to those in the training data.
- The 1,000-row sample file is synthetic. Its outcomes are drawn from the model's own probabilities, so scores on it
  only demonstrate how the app works.
""")
