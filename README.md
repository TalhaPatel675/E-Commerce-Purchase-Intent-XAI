# Purchase Intent XAI

Predict whether an online shopping session will end in a purchase, and explain every prediction.

This project trains and compares six classification models on 12,000 e-commerce sessions, picks the best one with leakage-safe cross-validation, chooses a business-driven decision threshold, explains the model with three methods (SHAP, permutation importance, LIME), audits its reliance on the `PageValues` feature, and serves it through a REST API and a Streamlit dashboard. Experiments are tracked and the final model is registered in MLflow.

> **Platform and Python version.** Developed and tested on **Windows with Python 3.12.10**. Use Python **3.10 to 3.12**. Do **not** use Python 3.13 or newer: the pinned packages do not support it (`pip install` fails on 3.14).

---

## 1. Results at a glance

| Item | Value |
|---|---|
| Selected model | **CatBoost** (chosen by cross-validated PR-AUC, not by test score) |
| Cross-validated PR-AUC (5-fold, training set) | **0.845 ± 0.009** |
| Held-out test PR-AUC (2,400 sessions) | **0.862** |
| Held-out test ROC-AUC | 0.971 |
| Decision threshold (business-driven) | **0.3335** (selected on out-of-fold predictions) |
| Test precision / recall at that threshold | 0.595 / 0.947 |
| Test PR-AUC **without** `PageValues` | **0.543** (CV 0.520 ± 0.021), a drop of about 0.32 |
| Positive rate (buyers) in the data | 15.7% |
| Registered MLflow model | `purchase-intent-best`, version 1 |
| Automated tests | 9 tests, all passing |

PR-AUC is the headline metric because only about 1 session in 6 ends in a purchase. A model that guesses at random scores a PR-AUC of roughly the positive rate (about 0.157), so 0.862 is far above chance, and even the model without `PageValues` (0.543) is well above chance.

---

## 2. Quick start (Windows)

You need: Windows 10/11, **Python 3.12** from <https://www.python.org/downloads/windows/> (tick *Add python.exe to PATH*), about 8 GB of RAM, and internet access for the first install.

Open a terminal in the project folder. In **PowerShell**, put `.\` in front of the script paths (for example `.\scripts\setup_windows.bat`).

```bat
:: 1. Create the virtual environment and install pinned dependencies (3-5 minutes)
scripts\setup_windows.bat

:: 2. Train everything: models, ablation, MLflow, SHAP, permutation, LIME, model card (1-2 minutes)
scripts\train.bat

:: 3. Start the API (leave this window open)  ->  http://127.0.0.1:8000/docs
scripts\run_api.bat

:: 4. In a second terminal, start the dashboard  ->  http://127.0.0.1:8501
scripts\run_dashboard.bat

:: 5. Optional: MLflow UI  ->  http://127.0.0.1:5000
scripts\run_mlflow.bat
```

**If you have several Python versions installed**, `setup_windows.bat` uses the first `python` on your PATH. Create the environment yourself first so it uses 3.12, then run the setup script (it skips creating the environment if `.venv` already exists):

```bat
py -3.12 -m venv .venv
scripts\setup_windows.bat
```

**The ZIP already contains trained models and reports**, so steps 3 and 4 work immediately after setup. The MLflow UI will be **empty until you run `scripts\train.bat`**, because the MLflow database stores absolute file paths and is therefore not shipped (see section 9).

---

## 3. Verification status

Everything below was checked by actually running it, not assumed.

**Verified** (Windows, Python 3.12.10, Docker Desktop 4.94):

- A fresh install from the final ZIP into an empty folder installs all pinned dependencies and passes all 9 tests.
- `black --check` and `ruff check` pass on `src`, `dashboard` and `tests`.
- All four notebooks execute end to end (run with `nbconvert`; their outputs are saved in the files). Re-running them needs `pip install nbconvert ipykernel` (or Jupyter), which is not in `requirements.txt`.
- Training runs end to end with `python -m src.models.train` (the command `train.bat` runs): six models, leakage-safe cross-validation, out-of-fold threshold, PageValues ablation, MLflow logging and registration, SHAP, permutation importance, LIME and the model card.
- The reproduced numbers match the ones in this README (CV PR-AUC 0.845, test PR-AUC 0.862, ablation test PR-AUC 0.543).
- MLflow: experiment `purchase-intent` with 8 runs and the registered model `purchase-intent-best` v1. The registered model loads from the registry and predicts from raw session data.
- API: `/health` and `/predict` work, with sensible probabilities and SHAP explanations. The confidence bands work.
- Dashboard: all six pages work (Overview, Live scoring, Funnel analytics, Model comparison, Explainability, Performance metrics).
- Docker: all three images build. The `api`, `dashboard` and `mlflow` services start and work, and the dashboard container reaches the API container at `http://api:8000`.

**Not verified. Do not rely on these without testing them yourself:**

- The `train` service inside Docker (a full `docker compose up --build`). It was deliberately not run so it would not overwrite the verified artefacts.
- `scripts\run_all.bat`, `scripts\lint.bat` and `scripts\test.bat` themselves (the underlying commands `python -m pytest`, `black` and `ruff` were run directly and pass).
- The GitHub Actions workflow itself (never run on GitHub; its lint, format and test commands pass locally).
- The AWS files in `aws/`.
- macOS and Linux. The Python code is portable, but the launch scripts are Windows `.bat` files and nothing was tested there.
- Python 3.10 and 3.11 (only 3.12.10 was tested).

---

## 4. The problem and the data

**Task.** Binary classification: given what happened in a shopping session, predict `Converted` (1 = purchase, 0 = no purchase).

**Data.** `dataset/ecommerce_sessions.csv` has 12,000 sessions and 18 columns: 1,886 purchases (15.7%) and 10,114 non-purchases. Its columns follow the widely used online-shoppers-intention layout. The repository does not document where the data came from, so confirm the source before citing one. Sessions fall in ten months (February, March, May to December); there are none in January or April.

| Column | Type | Meaning |
|---|---|---|
| `Administrative`, `Administrative_Duration` | count, seconds | Account/login-type pages visited, and time spent on them |
| `Informational`, `Informational_Duration` | count, seconds | Informational pages visited, and time spent on them |
| `ProductRelated`, `ProductRelated_Duration` | count, seconds | Product pages visited, and time spent on them |
| `BounceRates`, `ExitRates` | 0 to 1 | Average bounce and exit rate of the pages visited |
| `PageValues` | number | Average value of the pages visited; a very strong purchase signal (see section 7) |
| `SpecialDay` | 0 to 1 | Closeness to a special shopping day |
| `Month` | category | Month of the visit |
| `OperatingSystems`, `Browser`, `Region`, `TrafficType` | integer **IDs** | Treated as **categories**, not quantities (ID 4 is not "twice" ID 2) |
| `VisitorType` | category | Returning_Visitor / New_Visitor / Other |
| `Weekend` | boolean | Whether the session was on a weekend |
| `Converted` | target | 1 = purchase |

Full descriptions: `dataset/data_dictionary.csv`.

---

## 5. How the pipeline works

```
 ecommerce_sessions.csv (12,000 rows)
            |
            v
   Feature engineering  (5 new features)
            |
            v
   Stratified 80/20 split (seed 42)
     |                          |
     v                          v
 Training set (9,600)      Test set (2,400)  <- untouched until final evaluation
     |
     |  5-fold stratified cross-validation, six models.
     |  Inside EVERY fold: fit preprocessing on the fold's training part only,
     |  then apply it to the fold's validation part.
     v
   Pick the model with the best CV PR-AUC  ->  CatBoost
     |
     |  Out-of-fold (OOF) probabilities on the training set
     v
   Choose the business threshold from OOF predictions  ->  0.3335
     |
     v
   Refit on all training data, evaluate ONCE on the test set
     |
     v
   Explain (SHAP, permutation, LIME) + PageValues ablation + model card
     |
     v
   Log to MLflow and register a complete raw-input pipeline
```

### 5.1 Feature engineering (`src/features/engineer.py`)

Five behavioural features are added to the 17 raw columns:

| Feature | Definition |
|---|---|
| `TotalDuration` | Administrative + Informational + ProductRelated durations |
| `AvgProductDuration` | `ProductRelated_Duration / ProductRelated` (0 if there are no product pages), clipped to 0 to 20,000 |
| `BounceExitRatio` | `BounceRates / ExitRates` (0 if `ExitRates` is 0), capped at 5 |
| `PageValuePerProduct` | `PageValues / ProductRelated`, clipped to 0 to 200 (derived from `PageValues`) |
| `EngagementScore` | `ProductRelated + Informational + Administrative` (total pages viewed) |

### 5.2 Preprocessing (`src/preprocessing/preprocess.py`)

- **Categorical columns** (the seven listed above, including the integer IDs and `Weekend`): fill missing values with the most frequent value, then one-hot encode. Unseen categories are ignored (encoded as all zeros) instead of causing an error.
- **Numeric columns** (10 raw plus 5 engineered): fill missing values with the median, then apply `RobustScaler` (scales by median and inter-quartile range, so outliers matter less).
- After encoding, the model sees **80 columns**.
- The fitted preprocessor is saved to `models/preprocessing_pipeline.joblib`, so the API applies exactly the same transformation as training.

### 5.3 Class imbalance

Only 15.7% of sessions convert. The project uses **class weighting** (no resampling such as SMOTE, which would have to be done carefully inside each fold to avoid leakage). LogisticRegression, DecisionTree, RandomForest, LightGBM and CatBoost use balanced class weights. XGBoost is configured without class weighting.

### 5.4 The six models (`src/models/trainer.py`)

Fixed, reasonable settings are used. **No hyperparameter search was run**, so the scores below are not tuned.

| Model | Main settings |
|---|---|
| LogisticRegression | `C=1.0`, `max_iter=2000`, balanced class weight |
| DecisionTree | `max_depth=8`, balanced class weight |
| RandomForest | 300 trees, no depth limit, balanced class weight |
| XGBoost | 400 trees, depth 6, learning rate 0.05, subsample 0.9, column sample 0.9, `hist` |
| LightGBM | 400 trees, 31 leaves, learning rate 0.05, subsample 0.9, column sample 0.9, balanced class weight |
| CatBoost | 400 iterations, depth 6, learning rate 0.05, `Logloss`, balanced auto class weights |

### 5.5 How the models are evaluated

1. **Leakage-safe cross-validation.** The 9,600 training rows are split into 5 stratified folds. In each fold the preprocessing is fitted on that fold's training part only, then applied to its validation part. Fitting preprocessing on all the data before cross-validation would let information from validation rows leak into training, so this project does not do that. The folds come from scikit-learn's `StratifiedKFold` (shuffled, seed 42), so they are repeatable.
2. **Model selection uses CV PR-AUC only.** The test set plays no part in choosing the model.
3. **The test set is used only for reporting**: each model's test scores, the chosen model's evaluation at the business threshold, and the ablation's test score. No choice (model, threshold, features) is made from it.

### 5.6 Results for all six models

Sorted by the selection metric (CV PR-AUC). Test precision/recall/F1 in the model card use the default 0.5 cutoff; PR-AUC and ROC-AUC do not depend on a cutoff.

| Model | CV PR-AUC (mean ± std) | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|
| **CatBoost** | **0.845 ± 0.009** | **0.862** | 0.971 |
| LightGBM | 0.838 ± 0.006 | 0.839 | 0.967 |
| XGBoost | 0.837 ± 0.007 | 0.847 | 0.969 |
| RandomForest | 0.835 ± 0.012 | 0.849 | 0.969 |
| LogisticRegression | 0.804 ± 0.022 | 0.803 | 0.965 |
| DecisionTree | 0.742 ± 0.011 | 0.792 | 0.936 |

**An honest reading of this table:** CatBoost has the best CV score and the best test score, but its CV lead over LightGBM, XGBoost and RandomForest is only about 0.007 to 0.010, which is about the size of the fold-to-fold standard deviation. The four tree-ensemble models are close; CatBoost is a reasonable but not overwhelming winner. Logistic regression is a respectable baseline, and the single decision tree is clearly the weakest.

### 5.7 The business decision threshold

A model outputs a probability; the business needs a yes/no decision (for example "show a retention offer to this visitor"). The default cutoff of 0.5 ignores costs. This project chooses the cutoff by expected value (`src/utils/metrics.py`, `value_based_threshold`):

> expected value = (true positives x purchase value) minus (false positives x intervention cost)

The settings in `src/config/config.yaml` are `purchase_value: 5.0`, `intervention_cost: 1.0` and `min_recall: 0.60`. **These two money values are illustrative assumptions, not real business numbers**; replace them with your own and re-run training.

- The threshold is picked using **out-of-fold** predictions on the training set (predictions made by models that did not see those rows), so it is not tuned on data the model memorised.
- Result: threshold **0.3335**. On the untouched test set this gives precision **0.595** and recall **0.947**, with an expected value of **1,542** under the assumed values.
- Because a purchase is assumed to be worth five times the cost of an intervention, the best cutoff favours **catching nearly all buyers (high recall) at the price of many false alarms (lower precision)**. A different cost ratio gives a different threshold.

At the default 0.5 cutoff, CatBoost has test precision 0.657, recall 0.915 and F1 0.765.

---

## 6. Explainability: three lenses

All three are generated by `src/explain/shap_lime.py` during training.

| Method | Question it answers | Output |
|---|---|---|
| **SHAP** (global) | Which features push predictions up or down across many sessions, and by how much? | `reports/figures/shap_global_summary.png` |
| **Permutation importance** | How much does the model's PR-AUC fall if one feature's values are shuffled? | `reports/figures/permutation_importance.png` |
| **LIME** (local) | Why did the model give this one session its score? | `reports/lime_local_session0.html` |

**What they show.** All three agree that `PageValues` dominates.

| Rank | SHAP (mean absolute value) | Permutation importance (drop in PR-AUC) |
|---|---|---|
| 1 | PageValues 2.93 | PageValues 0.561 |
| 2 | PageValuePerProduct 1.49 | PageValuePerProduct 0.045 |
| 3 | ProductRelated 0.39 | ProductRelated 0.030 |
| 4 | EngagementScore 0.31 | ExitRates 0.025 |
| 5 | ExitRates 0.25 | BounceRates 0.017 |

**How to read them.**

- SHAP values for this boosted-tree model are in **log-odds** (the model's raw output), not probability points. Only the size and the direction (up or down) are meant to be compared.
- `PageValues` is far larger than everything else: removing its information hurts accuracy far more than removing any other feature (section 7).
- For tree models, SHAP uses the exact `TreeExplainer`.

**Caveats about the explanation artefacts.**

- They are computed on a random sample of 500 sessions drawn from the **whole dataset (training and test rows together)**, using the final model. They describe the model's behaviour, not its held-out performance.
- Global SHAP uses 300 of those 500 sessions.
- LIME explains one sample row (the first row of that sample, "session 0") on the encoded features (for example `Month_Feb`). It is a demonstration of local explanation, not a statistic about the whole data.

---

## 7. The PageValues audit (read this before the viva)

`PageValues` is by far the strongest feature. The question: **what if it is not available when you need the prediction?** (It is computed from how valuable the visited pages are, which may only be known late in or after the session, and it is closely tied to purchasing itself.) This is a deployment hazard, closely related to target leakage.

To measure the dependence, the project refits the winning model **without `PageValues` and without `PageValuePerProduct`** (the second feature is derived from the first, so keeping it would defeat the test), using the same leakage-safe cross-validation:

| | CV PR-AUC | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|
| With `PageValues` | 0.845 ± 0.009 | 0.862 | 0.971 |
| Without `PageValues` and `PageValuePerProduct` | 0.520 ± 0.021 | 0.543 | 0.841 |

The test PR-AUC falls by about **0.32**. The model without `PageValues` is still much better than guessing (0.543 versus 0.157), but it is a much weaker model.

**What this means:** the strong headline score rests on one feature. Whether that is acceptable depends on when `PageValues` is available in the real system. Treat it as a strong correlation, not a cause. Results: `reports/pagevalues_ablation.json`; write-up: `docs/pagevalues_ablation.md`.

---

## 8. Using the system

### 8.1 REST API (`src/api/main.py`)

Start it with `scripts\run_api.bat`. Interactive documentation: <http://127.0.0.1:8000/docs>.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Shows whether the model and preprocessing pipeline loaded, the model name and the threshold |
| `POST /predict` | Scores one session |
| `GET /` | Lists the endpoints |

**Example request** (PowerShell; a session that did convert in the data):

```powershell
$body = '{"session":{"Administrative":1,"Administrative_Duration":14.6517,"Informational":0,"Informational_Duration":0.0,"ProductRelated":19,"ProductRelated_Duration":283.8812,"BounceRates":0.008014,"ExitRates":0.042106,"PageValues":68.579222,"SpecialDay":0.0,"Month":"June","OperatingSystems":1,"Browser":2,"Region":1,"TrafficType":1,"VisitorType":"Returning_Visitor","Weekend":false}}'
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
```

**Example response** (shortened):

```json
{
  "prediction": 1,
  "conversion_probability": 0.909,
  "confidence": "high",
  "decision_threshold": 0.3335,
  "top_contributors": [
    {"feature": "PageValues", "direction": "increases_purchase_likelihood", "contribution": 3.85, "value": "pagevalues=68.579222"},
    {"feature": "PageValuePerProduct", "direction": "increases_purchase_likelihood", "contribution": 0.99, "value": null}
  ],
  "model_name": "CatBoost"
}
```

**Fields explained**

- `prediction`: 1 if `conversion_probability` is at least `decision_threshold` (0.3335), otherwise 0.
- `confidence`: how far the probability is from 0.5, ignoring which side: `high` if `max(p, 1-p) >= 0.85`, `medium` if it is at least 0.60, otherwise `low`. A probability of 0.05 is therefore "high" confidence (in "no purchase"). This is a simple band, not a calibrated confidence. The band limits are in `config.yaml`.
- `top_contributors`: the five features with the largest local SHAP values, with their direction and size (log-odds). The `value` field is a best-effort text match to the request and is `null` for engineered features.
- Inputs are validated (for example counts must be at least 0 and rates between 0 and 1). `Month` and `VisitorType` are free text; an unseen value is encoded as all zeros instead of failing.

### 8.2 Dashboard (`dashboard/app.py`)

Start it with `scripts\run_dashboard.bat` (the API must be running for *Live scoring*).

| Page | Shows |
|---|---|
| Overview | Session count, conversion rate, conversions by month |
| Live scoring | A form that sends a session to the API and shows the prediction and local SHAP bars |
| Funnel analytics | Conversion by product-page bucket, PageValues by visitor type, correlation heat map |
| Model comparison | Table and chart of all six models |
| Explainability | Global SHAP, permutation importance, LIME and top-feature tables |
| Performance metrics | Selected model, threshold, precision and recall, the full model card and the PageValues audit |

The API address defaults to `http://127.0.0.1:8000` and can be set with the `DASHBOARD_API_URL` environment variable. The *Live scoring* month list shows all twelve months, but the training data has none for January or April; those are encoded as all zeros.

### 8.3 MLflow (`scripts\run_mlflow.bat`, <http://127.0.0.1:5000>)

Training logs to a local SQLite database at `mlruns/mlflow.db`, experiment **`purchase-intent`**, producing **8 runs**:

- `cv-benchmark`: the parent run, with one nested run per model (six runs) holding that model's metrics;
- `BEST-CatBoost`: the final model's metrics (CV and test PR-AUC, threshold, precision and recall at the threshold, and the PageValues ablation result), plus the registered model.

**Registered model:** `purchase-intent-best` (version 1). It is a complete scikit-learn `Pipeline`: *feature engineering, then preprocessing, then CatBoost*. It takes a raw 17-column session table, so it needs no separate preprocessing step.

Things to know when using the registered model:

- It must be loaded from the **project root** (or with `PYTHONPATH=.`), because the pipeline refers to the project's own `FeatureEngineer` class in `src/features/engineer.py`.
- `predict()` on it returns **class labels (0/1)**, not probabilities. For probabilities, load it with `mlflow.sklearn.load_model("models:/purchase-intent-best/1")` and call `predict_proba`.
- The **business threshold (0.3335) is not part of the registered model**. It is applied in the API (`src/api/main.py`) and saved in `models/decision_threshold.json`.
- The logged signature's output type was inferred from probabilities, while `predict()` returns labels. This mismatch does not affect predictions, but be aware of it.

```bat
:: Example: load and score from the project root (with the virtual environment active)
python -c "import mlflow, pandas as pd; mlflow.set_tracking_uri('sqlite:///mlruns/mlflow.db'); m = mlflow.pyfunc.load_model('models:/purchase-intent-best/1'); df = pd.read_csv('dataset/ecommerce_sessions.csv').drop(columns=['Converted']).head(5); print(m.predict(df))"
```

### 8.4 Docker (optional)

Install Docker Desktop (Windows needs WSL2; if Docker says "WSL needs updating", run `wsl --update` in an administrator terminal).

```bat
docker compose build
docker compose up -d --no-deps api dashboard mlflow
```

This starts the API (8000), dashboard (8501) and MLflow UI (5000) using the artefacts already in `models/`, `reports/` and `mlruns/`. Stop everything with `docker compose down`.

Plain `docker compose up --build` **also runs the `train` service first** and retrains before the API starts. That path was **not tested** (see section 3). If you run it against an `mlruns` folder created on another machine, MLflow may try to write to paths from that machine; start from an empty `mlruns` folder to be safe.

---

## 9. Repository layout

```
purchase-intent-xai/
├── README.md
├── requirements.txt          pinned dependencies
├── pyproject.toml
├── docker-compose.yml        train, api, dashboard, mlflow services
├── dataset/                  ecommerce_sessions.csv, data_dictionary.csv
├── src/
│   ├── config/               config.yaml (all settings) and the loader
│   ├── data/                 data loader and validation
│   ├── features/             feature engineering
│   ├── preprocessing/        encoding and scaling
│   ├── models/               train.py (pipeline), trainer.py (models), ablation.py
│   ├── explain/              SHAP, permutation importance, LIME, model card
│   ├── api/                  FastAPI app and request/response schemas
│   └── utils/                metrics (including the threshold search), logging
├── dashboard/app.py          Streamlit dashboard
├── models/                   trained model, preprocessing pipeline, threshold, metadata
├── reports/                  model card, figures, LIME HTML, ablation JSON, explainability summary
├── mlruns/                   MLflow database (created by training; empty in the ZIP)
├── tests/                    9 automated tests
├── scripts/                  8 Windows .bat launchers
├── docker/                   Dockerfile.api, Dockerfile.dashboard, Dockerfile.train
├── docs/                     architecture, install guide, user guide, ablation write-up, more
├── notebooks/                four notebooks, executed end to end with outputs saved in the files
├── aws/                      deployment notes and a task-definition template (not tested)
└── .github/workflows/ci.yml  CI workflow (never run on GitHub; its lint and test commands pass locally)
```

**Why the ZIP has no `mlflow.db`:** MLflow stores absolute paths (for example `C:\projects\...`) for model artefacts. A database created on one machine would point to folders that do not exist on another. Running `scripts\train.bat` creates a correct one anywhere.

---

## 10. Configuration

Almost everything is set in `src/config/config.yaml`: file paths, which columns are numeric or categorical, the test size (0.20), the random seed (42), the number of folds (5), the money values for the threshold, the confidence bands, the API and dashboard ports, and the MLflow experiment and model names. Change a value there and re-run training; no code changes are needed. `.env.example` lists optional environment settings; the code reads its main settings from `config.yaml` (the dashboard also honours `DASHBOARD_API_URL`).

---

## 11. Tests

```bat
set PYTHONPATH=.
python -m pytest -q
```

Nine tests (4 in `tests/test_data.py`, 5 in `tests/test_predict.py`) check: loading and validating the real dataset, the feature engineering, the preprocessing output shape, the handling of unseen categories, the `/health` and `/predict` endpoints, and the confidence bands. The two API tests are skipped if the trained artefacts are missing, so run `scripts\train.bat` first. In PowerShell use `$env:PYTHONPATH="."`.

---

## 12. Reproducibility

- A fixed random seed (42), a fixed stratified split, and fixed CV folds.
- Every dependency is pinned in `requirements.txt`, including the database packages MLflow needs (`SQLAlchemy==2.0.36`, `alembic==1.14.0`) and `httpx==0.27.2` for the API tests.
- Retraining on the dev machine reproduced the same results (CV PR-AUC 0.845, test PR-AUC 0.862, threshold 0.3335). Small differences in the last digits are possible on other machines or library versions.

---

## 13. Concepts in plain language

| Term | Meaning |
|---|---|
| **PR-AUC** | Area under the precision-recall curve. Precision is "of the sessions I flag, how many really buy?"; recall is "of all real buyers, how many do I catch?". It is the right metric when positives are rare, because accuracy would look high (84%) for a model that always says "no purchase". |
| **ROC-AUC** | The chance that a random buyer is ranked above a random non-buyer. Less sensitive to rare positives than PR-AUC. |
| **Cross-validation** | Split the training data into 5 parts; train on 4, score on the 5th, rotate, average. Gives a more stable estimate than one split. |
| **Data leakage** | Letting information from validation or test data influence training (for example scaling using statistics from all rows). It makes scores look better than they will be in real use. |
| **Out-of-fold (OOF) prediction** | A prediction for a row made by a model that was trained without that row. Lets you tune choices such as the threshold honestly. |
| **Decision threshold** | The probability above which you act. 0.5 is only a default; the right value depends on costs. |
| **SHAP** | Splits a prediction into a contribution from each feature (based on game theory), for one prediction or averaged over many. |
| **Permutation importance** | Shuffle one feature and see how much the score falls. A big fall means the model relies on it. |
| **LIME** | Explains one prediction by fitting a small, simple model around that single session. |
| **Ablation** | Remove a part (here, `PageValues`) and measure how much worse the result gets. |
| **MLflow model registry** | A versioned catalogue of models, so a specific version can be loaded by name. |

---

## 14. Responsible use, deviations from the PRD, and known limitations

**Responsible use.** The data is anonymised session behaviour. Use the model for aggregate funnel analysis and to decide which sessions might receive a low-cost nudge (for example a discount banner). Do **not** use it to identify or profile a named person, to deny service, to set an individual's price, or for any decision with legal or financial consequences for a person, and do not join the data with personally identifying fields. `PageValues` may not be known when a visitor is first seen, so do not score early in a session without re-checking the audit in section 7. The same statement is in `reports/MODEL_CARD.md`.

**Deviations from the PRD** (recorded here, as the PRD's document-control section invites):

- **No hyperparameter tuning.** PRD section 4.5 asks to tune the most promising models. The six models use fixed settings (section 5.4).
- **Imbalance strategy not benchmarked.** Class weighting was chosen and documented. SMOTE and threshold-only tuning were not compared (a PRD hint, not a requirement).
- **No feature-removal step.** PRD section 4.4 asks to remove features that do not earn their place. Importance is reported with two methods, but all features are kept (`selection_method: "none"` in `config.yaml`).
- **No precision-recall-curve plot.** Accuracy and the confusion matrix are stored in `models/training_metadata.json` but are not shown in the dashboard or the model card.
- **SHAP explainer built per request.** The API creates the SHAP explainer on every `/predict` call instead of once at startup (the model itself loads once).
- **Not a Git repository, and CI never ran.** The submission is a ZIP. `.github/workflows/ci.yml` is included but was never run on GitHub.
- **Stretch goals not done:** probability-calibration curve, data-drift check, live AWS deployment. (The expected-value threshold and the PageValues ablation are done.)

**Other limitations**

- **One strong feature.** The performance depends heavily on `PageValues` (section 7). Check whether it is available at prediction time before trusting the headline score.
- **Illustrative costs.** The threshold uses assumed values (purchase = 5, intervention = 1). It is a method demonstration, not a business result.
- **No tuning.** Hyperparameters are fixed defaults; models might improve with a search.
- **Close competition.** The top four models are within about one standard deviation of each other in cross-validation.
- **Explanation samples.** SHAP, permutation importance and LIME use a random sample from the whole dataset, including rows the model trained on.
- **Data provenance.** The source of the dataset is not documented in this repository.
- **Demo-grade API.** There is no authentication, and cross-origin requests are open to all origins. The SHAP explainer is rebuilt on every request, which is fine for a demo but slow at scale. The confidence band is not calibrated. A reliability (calibration) curve is not implemented.
- **Unverified parts.** See section 3.

---

## 15. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `pip install` fails building pandas (`Could not find vswhere.exe`, `metadata-generation-failed`) | Wrong Python version (3.13 or newer). Install Python 3.12 and create the environment with `py -3.12 -m venv .venv`. |
| `.\scripts\setup_windows.bat` is "not recognised" | In PowerShell the scripts need `.\` in front, and they live in the `scripts` folder. |
| `cannot import name 'FallbackAsyncAdaptedQueuePool'` | SQLAlchemy is too new for MLflow 2.18. The pinned `SQLAlchemy==2.0.36` avoids this; reinstall with `pip install -r requirements.txt`. |
| API returns 503 "Model not loaded" | Trained artefacts are missing. Run `scripts\train.bat`. |
| Dashboard *Live scoring* shows a connection error | The API is not running, or the API URL box does not match it. Start `scripts\run_api.bat`. |
| MLflow UI shows no runs or models | Training has not been run on this machine. Run `scripts\train.bat`, then `scripts\run_mlflow.bat`. |
| PowerShell blocks activating `.venv` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`, then activate again. |
| Loading the registered model fails with `No module named 'src'` | Run from the project root, or set `PYTHONPATH=.`. |
| Docker: "WSL needs updating" | Run `wsl --update` in an administrator terminal, then restart Docker Desktop. |
| Docker: `docker` not recognised in a terminal | Docker was installed after the terminal opened. Close VS Code or the terminal completely and reopen it. |

---

## 16. More documentation

| File | Content |
|---|---|
| `reports/MODEL_CARD.md` | Generated model card: intended use, metrics, importance tables, limitations |
| `docs/pagevalues_ablation.md` | Full write-up of the PageValues audit |
| `docs/architecture.md` | Architecture notes |
| `docs/user_guide.md` | Using the API and dashboard |
| `docs/install_windows.md` | Detailed Windows installation |
| `docs/requirements.md`, `docs/rubric_map.md` | Requirements and how they map to the project |
