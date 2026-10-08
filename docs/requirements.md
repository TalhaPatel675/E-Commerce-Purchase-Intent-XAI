# Requirements traceability

Every PRD requirement and the file/line that fulfils it.

## §1.2 "Done looks like"

| # | Requirement | Implementation |
|---|---|---|
| 1 | Reproducible training → saved versioned model + preprocessing | `src/models/train.py`, `models/best_model.joblib`, `models/preprocessing_pipeline.joblib` |
| 2 | Honest model-comparison report across six algorithms using imbalance-aware metrics | `src/models/train.py:evaluate_models`, `models/training_metadata.json`, `reports/MODEL_CARD.md` |
| 3 | Decision threshold from business value, not 0.5 | `src/utils/metrics.py:value_based_threshold`, `models/decision_threshold.json` |
| 4 | Global SHAP + local SHAP + LIME + model card | `src/explain/shap_lime.py`, `reports/figures/*.png`, `reports/lime_local_session0.html`, `reports/MODEL_CARD.md`, `src/api/main.py:_explain_row` |
| 5 | FastAPI: prob + decision + confidence + top contributors | `src/api/main.py`, `src/api/schemas.py:PredictResponse` |
| 6 | Streamlit dashboard (funnel, live scoring, compare, explain) | `dashboard/app.py` (six sections) |
| 7 | Docker + Compose + MLflow + GitHub Actions + AWS-ready | `docker/`, `docker-compose.yml`, `mlruns/`, `.github/workflows/ci.yml`, `aws/` |
| 8 | Documentation a stranger can follow | `README.md`, `docs/install_windows.md`, `docs/user_guide.md` |

## §4.1 Data ingestion & configuration

| Requirement | Implementation |
|---|---|
| Loader reads the CSV, validates schema, raises clear error | `src/data/loader.py:DataValidationError` + `tests/test_data.py::test_validation_missing_target` |
| Single config source | `src/config/config.yaml` + `src/config/settings.py` |
| Reproducible split | `random_seed: 42` + `train_test_split(..., stratify=y)` |
| Type hints + logging module + exceptions | every src/ file |

## §4.2 Data preprocessing

| Requirement | Implementation |
|---|---|
| Single reusable ColumnTransformer | `src/preprocessing/preprocess.py` |
| Encode int IDs (OS/Browser/Region/TrafficType) as categorical | `categorical_columns` in `config.yaml` |
| Scaling per fold, never on full data | `pipeline.fit_transform(X_train_df)` then `pipeline.transform(X_test_df)` |
| Fitted pipeline saved | `models/preprocessing_pipeline.joblib` (loaded at API startup, never re-fit) |
| Imbalance strategy documented | `class_weight="balanced"` (`src/models/trainer.py`). SMOTE and threshold-only tuning were NOT benchmarked (recorded as a deviation in README section 14) |

## §4.3 EDA

| Requirement | Implementation |
|---|---|
| Conversion behaviour, seasonality, correlations | `notebooks/01_eda.ipynb` |
| One interactive Plotly visual | `notebooks/01_eda.ipynb` (`px.bar`) + `dashboard/app.py:section_funnel` |
| Stated takeaways per figure | Notebook markdown cells + `reports/MODEL_CARD.md` |

## §4.4 Feature engineering & selection

| Requirement | Implementation |
|---|---|
| At least a few justified engineered features | 5 features in `src/features/engineer.py`; rationale in `dashboard` and `notebooks/02_features.ipynb` |
| Feature importance by **more than one method** | SHAP (`src/explain/shap_lime.py:generate_global_shap`) + scikit-learn permutation importance (`generate_permutation_importance`); both figures in `reports/figures/` |
| Final feature set explicitly documented | `notebooks/02_features.ipynb` MD cells + `docs/user_guide.md` table |

## §4.5 Modelling

| Requirement | Implementation |
|---|---|
| Six algorithms train through common interface | `src/models/trainer.py:default_model_factory` + `boosting_factory`; winner selected by leakage-safe CV PR-AUC |
| Full metric suite with leakage-safe CV PR-AUC, recall foremost | `src/utils/metrics.py:full_metric_suite`, leakage-safe fold-local preprocessing in `src/models/train.py`, `cv_pr_auc_mean/std` |
| Value-based decision threshold selected from OOF predictions (not 0.5) | `models/decision_threshold.json` + MODEL_CARD justification; threshold selected on out-of-fold training predictions and evaluated on held-out test data |
| Best model + preprocessing persisted | `models/best_model.joblib`, `models/preprocessing_pipeline.joblib` |

## §4.6 Explainable AI

| Requirement | Implementation |
|---|---|
| Global SHAP summary | `reports/figures/shap_global_summary.png` |
| Local SHAP for any single session | `src/api/main.py:_explain_row` (TreeExplainer for trees per §4.6 hint) |
| LIME per-session explanation | `reports/lime_local_session0.html`, generated via `generate_local_lime` |
| Model card documents intended use + limitations | `reports/MODEL_CARD.md` (Intended use, Selected model, Threshold, Performance, Importance ×2 lenses, PageValues audit, Limitations) |
| Explanations surfaced in API and dashboard | API: `PredictResponse.top_contributors`; Dashboard: dedicated Explainability section shows both PNGs + LIME HTML |

## §4.7 Real-time API

| Requirement | Implementation |
|---|---|
| POST `/predict` returns prediction + prob + confidence + contributors | `src/api/main.py:predict`, `src/api/schemas.py:PredictResponse` |
| 422 on bad input via Pydantic | `SessionFeatures` Field constraints |
| Unknown categories handled | `OneHotEncoder(handle_unknown="ignore")`; `tests/test_predict.py::test_unknown_category_handled` |
| GET `/health` | `src/api/main.py:health` |
| Load once at startup | `src/api/main.py:_load_artifacts()` import-time |
| Interactive docs at `/docs` | FastAPI default |

## §4.8 Dashboard

| Requirement | Implementation |
|---|---|
| Six sections | `dashboard/app.py:PAGES` dict |
| Live session scoring returns result with explanation | `section_live_scoring` calls `requests.post("/predict")` |
| Interactive Plotly charts | `px.histogram`, `px.bar`, `px.imshow`, `px.box` |
| Never retrain on load | `@st.cache_data` on data loaders; model lives in the API service |

## §4.9 Deployment & MLOps

| Requirement | Implementation |
|---|---|
| `docker-compose up` brings API + dashboard locally | `docker-compose.yml` |
| MLflow shows experiment runs + registered model | SQLite-backed MLflow tracking store; best raw-input feature+preprocessing+model pipeline is registered as `purchase-intent-best` |
| CI workflow runs lint + test + image build green | `.github/workflows/ci.yml` (multi-Python matrix + docker build on main) |
| AWS deployment steps documented + config parameterised | `aws/deploy_notes.md` + `aws/task-definition-template.json` |
| `.dockerignore` | root `.dockerignore` |
| `libgomp1` for LightGBM | installed in every Dockerfile; reminder in `aws/deploy_notes.md` and `docs/install_windows.md` |
| Non-root user in API/Dashboard images | Dockerfiles create `appuser` |

## §5 Engineering standards

OOP where it earns its keep (DataLoader, PreprocessingPipeline, FeatureEngineer, Trainer); type hints on every public function; logging module via `src/utils/logger.py:configure_logging`; explicit `DataValidationError` / `RuntimeError`; black + ruff configured (`pyproject.toml`); pytest in `tests/`; deterministic seeds + pinned deps + no secrets.

## §9 Submission deliverables

Repo structure per §7 ✅ · dataset + dictionary ✅ · four notebooks named after CRISP-DM phases ✅ · reusable `src/` ✅ · saved model + preprocessing artefacts ✅ · MLflow run log + registered model (created by `scripts\train.bat`; the ZIP ships an empty `mlruns/`) · model card ✅ (`reports/MODEL_CARD.md`) · FastAPI + Streamlit runnable via docker-compose ✅ · GitHub Actions workflow included (never run on GitHub) · README + diagrams + install + user + model docs ✅ · PageValues ablation write-up ✅.

The 3-5 minute demo recording is the only deliverable you must produce yourself (it's a screen capture of the running API + dashboard).

## §10 Rubric

See [`docs/rubric_map.md`](rubric_map.md).

## §12 Environment

Python 3.10+ venv created by `scripts\setup_windows.bat`. All §12.1 + §12.2 pins live in `requirements.txt`. The Docker base uses slim Python 3.12 + libgomp1 per the §4.9 hint.
