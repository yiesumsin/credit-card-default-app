# 💳 Credit Card Default Predictor

A web app that estimates the chance that a credit card client misses a payment next month. It uses the
**Gradient Boosted Trees** model trained in **KNIME** (exported as a PMML file) and shows the result with cards,
a submitted assessment, and interactive Plotly charts. This README explains, step by step, how to put it online **for free** using
GitHub and Streamlit Community Cloud. No earlier GitHub experience is needed.

> **Proof of concept.** The model is trained on public data from Taiwan (2005), not on EPF data, and is not meant for real
> credit decisions. It shows the method.

---

## 1. What is in this folder

To run locally on this computer, double-click `run_app.bat`. It uses the project's `.venv` environment,
where Streamlit and PyPMML are installed. If an older app is running in a terminal, stop it with Ctrl+C first.
Alternatively, from PowerShell in this folder, run:

```powershell
& '.venv/Scripts/python.exe' -m streamlit run app.py
```

If `No module named 'pypmml'` appears after using a different launch command, that command may be using another
Python environment. Use the launcher above rather than a bare `streamlit run app.py` command.

| File or folder | What it does |
| --- | --- |
| `app.py` | The app itself: page layout, client form, the four tabs |
| `credit_features.py` | Turns the raw client columns into the 16 model inputs, with the same formulas as KNIME |
| `pmml_model.py` | Loads the PMML file and returns a probability of default |
| `ui.py` | Colours (forest green, slate, white), custom CSS and all the Plotly charts |
| `metrics.py` | Confusion matrix, ROC curve and AUC calculations |
| `credit_card_default_classifier.pmml` | **The trained model** exported from KNIME |
| `model_meta.json` | Your KNIME cross-validation results and the cutoff (0.30) shown on the Model performance tab |
| `reference_roc.csv` | Points used to draw the ROC curve |
| `sample_data/synthetic_clients_1000.csv` | 1,000 made-up clients **with** outcomes (for testing and presenting) |
| `sample_data/synthetic_clients_1000_no_labels.csv` | The same clients **without** the outcome column |
| `generate_synthetic_data.py` | Script that created the sample files (you do not need to run it) |
| `requirements.txt` | The Python packages the app needs |
| `packages.txt` | Tells the hosting service to install **Java** (the PMML model needs it) |
| `.streamlit/config.toml` | The app's colour theme |

---

## 2. The app in one minute

* **Client assessment:** enter the profile, then six monthly rows containing repayment status, bill amount and amount paid.
  Type exact amounts in NT$. Use **Try an example client** to fill the form, then select **Calculate risk**.
  Results show the submitted client's probability, the dataset average and the current cutoff. Edits take effect only
  when you submit again. Loading an example clears the previous assessment.
* **Decision cutoff:** below the client form and above its result, change the decision cutoff for both individual and batch scoring (default **0.30**).
* **Assessment result:** a restrained risk summary, a "what if" repayment-status chart, and the 16 model inputs.
  The submitted client values are also available for review.
* **Score a CSV file:** upload a file of clients. You get the risk for every client, charts, and a
  download button. If the file has a `default` column (1 = defaulted, 0 = did not), you also get the confusion matrix and
  the ROC curve.
* **Model performance:** the KNIME cross-validation results (ROC AUC, recall, precision, F1, confusion matrix, ROC curve).
* **About:** how the model and inputs work.

### Format of a CSV you upload
One row per client, with these columns (any order; extra columns such as `ID` are kept):

`LIMIT_BAL, AGE, EDUCATION, MARRIAGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1 ... BILL_AMT6, PAY_AMT1 ... PAY_AMT6`

Optional: `default` (0 or 1). A column called `PAY_1` is accepted in place of `PAY_0`. `SEX` is not needed. `MARRIAGE` is required: 1 = married, 2 = single, 3 = others. Undocumented codes must be resolved before upload; they are not silently recoded.


## 3. Using a different model later

1. In KNIME, retrain and export the model with **Gradient Boosted Trees to PMML** and the **PMML Writer**. Replace
   `credit_card_default_classifier.pmml` (same name).
2. The model's input names must stay `LIMIT_BAL, AGE, PAY_0, avg_bill, total_paid, max_delay, utilisation, pay_ratio, n_late,
   EDUCATION_1 ... EDUCATION_4`. If you add, remove or change an engineered column, change `MODEL_COLUMNS` and `add_features()`
   in `credit_features.py` to match your KNIME formulas.
3. Update the results in `model_meta.json` and the cutoff, and re-run `python generate_synthetic_data.py` if you want new sample
   files (it needs Java).


---

## 4. Good to know

* **Cutoff 0.30.** A client is flagged when the probability is at or above the cutoff. A lower cutoff catches more defaulters
  and raises more false alarms.
* **Probabilities.** The model was trained without SMOTE, so its probabilities are close to real default rates. The average
  client is about 22%.
* **Privacy.** Uploaded CSV files are processed in the app session and are not saved by this app, but the hosted app is public.
  Do not upload confidential data to it.
* **Dropped columns.** Sex is not used; marital status is one-hot encoded. The monthly bill, payment and status columns were replaced by the
  summary columns, except the latest repayment status (`PAY_0`), which carries information the summaries do not.


