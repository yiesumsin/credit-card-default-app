# 💳 Credit Card Default Predictor

A web app that estimates the chance that a credit card client misses a payment next month. It uses the
**Gradient Boosted Trees** model trained in **KNIME** (exported as a PMML file) and shows the result with cards,
a gauge, and interactive Plotly charts. This README explains, step by step, how to put it online **for free** using
GitHub and Streamlit Community Cloud. No earlier GitHub experience is needed.

> **Proof of concept.** The model is trained on public data from Taiwan (2005), not on EPF data, and is not meant for real
> credit decisions. It shows the method.

---

## 1. What is in this folder

| File or folder | What it does |
| --- | --- |
| `app.py` | The app itself: page layout, sidebar inputs, the four tabs |
| `credit_features.py` | Turns the raw client columns into the 13 model inputs, with the same formulas as KNIME |
| `pmml_model.py` | Loads the PMML file and returns a probability of default |
| `ui.py` | Colours (purple, blue, white), custom CSS and all the Plotly charts |
| `metrics.py` | Confusion matrix, ROC curve and AUC calculations |
| `credit_card_default_classifier.pmml` | **The trained model** exported from KNIME |
| `model_meta.json` | Your KNIME cross-validation results and the cutoff (0.25) shown on the Model performance tab |
| `reference_roc.csv` | Points used to draw the ROC curve |
| `sample_data/synthetic_clients_1000.csv` | 1,000 made-up clients **with** outcomes (for testing and presenting) |
| `sample_data/synthetic_clients_1000_no_labels.csv` | The same clients **without** the outcome column |
| `generate_synthetic_data.py` | Script that created the sample files (you do not need to run it) |
| `requirements.txt` | The Python packages the app needs |
| `packages.txt` | Tells the hosting service to install **Java** (the PMML model needs it) |
| `.streamlit/config.toml` | The app's colour theme |

---

## 2. The app in one minute

* **Sidebar:** enter a client's details with sliders and dropdowns (credit limit, age, education, six months of repayment
  status, bills and payments). Three **Quick example** buttons fill the whole form. A **Decision cutoff** slider lets you
  change when a client is flagged (the chosen cutoff is **0.25**).
* **Predict one client:** a red box (high risk) or green box (low risk), the probability, a gauge, a "what if the latest
  status were different?" chart, and the 13 values the model actually sees.
* **Score a CSV file:** upload a file of clients (or use the sample file). You get the risk for every client, charts, and a
  download button. If the file has a `default` column (1 = defaulted, 0 = did not), you also get the confusion matrix and
  the ROC curve.
* **Model performance:** the KNIME cross-validation results (ROC AUC, recall, precision, F1, confusion matrix, ROC curve).
* **About:** how the model and inputs work, and the limits.

### Format of a CSV you upload
One row per client, with these columns (any order; extra columns such as `ID` are kept):

`LIMIT_BAL, AGE, EDUCATION, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1 ... BILL_AMT6, PAY_AMT1 ... PAY_AMT6`

Optional: `default` (0 or 1). A column called `PAY_1` is accepted in place of `PAY_0`. `SEX` and `MARRIAGE` are not needed.

---

## 3. Put the app online for free (GitHub + Streamlit Community Cloud)

You need two free accounts: **GitHub** (stores your files) and **Streamlit Community Cloud** (runs the app, and signs in with
your GitHub account). The screens change from time to time, so the button names below may differ slightly.

> **Before you start: GitHub repositories used for free hosting are public.** This project contains only the public dataset's
> model, made-up sample clients and code, so that is fine. **Never upload real or confidential EPF data to it.**

### Step 1: Create a GitHub account
1. Go to <https://github.com> and click **Sign up**.
2. Follow the steps (email, password, a username, email verification). Remember your username.

### Step 2: Create a repository (a folder on GitHub)
1. After signing in, click the **+** at the top right and choose **New repository**.
2. **Repository name:** `credit-default-app` (any name without spaces).
3. Choose **Public**.
4. Leave "Add a README file" **unticked** (this folder already has one).
5. Click **Create repository**.

### Step 3: Upload the files
1. **Unzip** the zip file on your computer first. You will get a folder called `credit_default_app`.
2. On your new empty repository page, click the link **uploading an existing file**.
3. Open the `credit_default_app` folder on your computer and **select everything inside it** (not the folder itself), then
   drag it into the GitHub page. Folders such as `sample_data` upload with their files.
4. **Important:** `app.py` must end up at the top level of the repository, not inside an extra folder.
5. Wait until all files show as uploaded, then click the green **Commit changes** button at the bottom.
6. **Check the hidden folder.** On GitHub, look for a folder named `.streamlit`. If it is missing (some computers hide folders
   that start with a dot):
   1. Click **Add file → Create new file**.
   2. In the name box type exactly `.streamlit/config.toml` (typing the `/` creates the folder).
   3. Open `.streamlit/config.toml` from the unzipped folder in a text editor, copy everything and paste it into GitHub.
   4. Click **Commit changes**.
   The app still works without this file, because `ui.py` also sets the colours, but the sliders will not be purple.
7. Also check that `packages.txt` is in the repository. It contains one line, `default-jre-headless`, and it is what installs Java.

### Step 4: Deploy on Streamlit Community Cloud
1. Go to <https://share.streamlit.io> and click **Continue with GitHub** (or **Sign in**). Allow Streamlit to access your GitHub.
2. Click **Create app** (or **New app**), then choose **Deploy a public app from GitHub**.
3. Fill in:
   * **Repository:** `your-username/credit-default-app`
   * **Branch:** `main`
   * **Main file path:** `app.py`
   * **App URL:** pick a short name (optional)
4. Open **Advanced settings** and choose **Python 3.12** if the option is there (3.11 also works).
5. Click **Deploy**.
6. The first start takes **2 to 5 minutes**, because it installs Java and the packages. You can watch progress in the log on the
   right ("Manage app" at the bottom right opens it later).
7. When the app appears, **copy the web address** (it ends in `.streamlit.app`) and share it.

Free apps go to sleep after a period with no visitors. The next visitor sees a "Wake up" button and the app starts again in a
minute. **Open it a few minutes before a presentation** so it is awake.

### Step 5: Changing something later
* **To edit a file:** open it on GitHub, click the pencil icon, edit, then **Commit changes**. The app updates by itself in
  about a minute.
* **To replace a file** (for example a new PMML): on the repository page click **Add file → Upload files**, drag the new file
  with the **same name**, and commit.
* **To restart the app:** on share.streamlit.io, open the app's menu (three dots) and choose **Reboot**.

---

## 4. Run it on your own computer (optional)

1. Install **Python 3.10 or newer** from <https://www.python.org> and a **Java runtime (version 8 or newer)**, for example
   from <https://adoptium.net>. Check Java with `java -version` in a terminal.
2. Open a terminal **inside this folder** and run:
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Your browser opens at `http://localhost:8501`. Press `Ctrl + C` in the terminal to stop.

---

## 5. Suggested 5-minute walkthrough for your presentation

1. **Predict tab.** Click **Low risk client** (about 12% probability, green), then **High risk client** (about 65%, red).
   Click **Borderline client** (about 27%, just over the 25% cutoff) and drag the **Decision cutoff** slider to show how the
   decision flips. Point out the "what if" chart: the latest repayment status is the strongest driver.
2. **Score a CSV tab.** Choose "sample file with outcomes", show the risk distribution, the flagged clients, and download the
   results. Scroll to the confusion matrix and ROC curve, and say that the outcomes in this file are synthetic.
3. **Model performance tab.** This is the honest evidence: 5-fold cross-validated **ROC AUC 0.779**, **recall 62.6%** and
   **precision 47.1%** at the 0.25 cutoff. Show how SMOTE and the single split compared.
4. **About tab.** Explain the 13 inputs, the dropped columns and the limits.

---

## 6. Using a different model later

1. In KNIME, retrain and export the model with **Gradient Boosted Trees to PMML** and the **PMML Writer**. Replace
   `credit_card_default_classifier.pmml` (same name).
2. The model's input names must stay `LIMIT_BAL, AGE, PAY_0, avg_bill, total_paid, max_delay, utilisation, pay_ratio, n_late,
   EDUCATION_1 ... EDUCATION_4`. If you add, remove or change an engineered column, change `MODEL_COLUMNS` and `add_features()`
   in `credit_features.py` to match your KNIME formulas.
3. Update the results in `model_meta.json` and the cutoff, and re-run `python generate_synthetic_data.py` if you want new sample
   files (it needs Java).

**Formulas used by `credit_features.py` (identical to the KNIME workflow):**
```
avg_bill    = (BILL_AMT1 + ... + BILL_AMT6) / 6
total_paid  = PAY_AMT1 + ... + PAY_AMT6
max_delay   = maximum of PAY_0, PAY_2 ... PAY_6
utilisation = BILL_AMT1 / LIMIT_BAL
pay_ratio   = total_paid / (bill total, or 1 if the bill total is 0)
n_late      = number of PAY_ columns greater than 0
EDUCATION   = codes 0, 5, 6 recoded to 4, then one-hot into EDUCATION_1 ... EDUCATION_4
```

---

## 7. Troubleshooting

| Problem | What to do |
| --- | --- |
| "The model could not be loaded" or a Java error | Java is missing. On Streamlit Community Cloud check that `packages.txt` is in the repository and reboot the app. On your computer install Java 8 or newer. |
| `ModuleNotFoundError` in the log | A package is missing from `requirements.txt`. Add its name on a new line, commit, and the app rebuilds. |
| "File does not exist: app.py" when deploying | The files were uploaded inside an extra folder. Move them to the top level of the repository, or set **Main file path** to `folder_name/app.py`. |
| The app shows "Oh no" or a red error box | Open **Manage app** (bottom right) and read the log. The last lines say what failed. |
| Sliders are not purple | `.streamlit/config.toml` is missing. See step 3.6. |
| "The file is missing these columns" after uploading | Check the column names against section 2. Names are case-sensitive. |
| Rows were "left out" with a warning | Those rows had empty or non-numeric values, or a zero credit limit. Fix them in the CSV. |
| The app is slow on first open | The free app was asleep. Wait a minute. |
| A very large CSV fails | Free hosting has limited memory. Try fewer than about 50,000 rows. |
| GitHub says a file is too large | The browser upload limit is 25 MB per file. All files here are far below it. |

---

## 8. Good to know

* **Cutoff 0.25.** A client is flagged when the probability is at or above the cutoff. A lower cutoff catches more defaulters
  and raises more false alarms.
* **Probabilities.** The model was trained without SMOTE, so its probabilities are close to real default rates. The average
  client is about 22%.
* **Privacy.** Uploaded CSV files are processed in the app session and are not saved by this app, but the hosted app is public.
  Do not upload confidential data to it.
* **Dropped columns.** Sex and marital status are not used. The monthly bill, payment and status columns were replaced by the
  summary columns, except the latest repayment status (`PAY_0`), which carries information the summaries do not.
