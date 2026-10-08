# Architecture (PRD §6)

```
                            ┌──────────────────────────┐
                            │  Streamlit dashboard      │
                            │  dashboard/app.py (6 sec) │
                            └──────────┬───────────────┘
                                       │ HTTP POST /predict
                                       ▼
                            ┌──────────────────────────┐
                            │   FastAPI service        │
                            │   src/api/main.py        │
                            │   + Pydantic + SHAP Tree │
                            └──────────┬───────────────┘
                                       │ joblib.load (load-once)
                                       ▼
            ┌────────────────────┐  pipeline  ┌────────────────────┐
            │  ML layer          │  joblib    │ Trained model      │
            │  6 algorithms      │  load      │ XGBoost / LGBM /   │
            │  StratKFold CV     │ ───────────│ CatBoost / RF / ...│
            │  SHAP + Perm + LIME│            │ best_model.joblib  │
            └────────┬───────────┘            └─────────┬──────────┘
                     └────────────────┬─────────────────┘
                                      ▼
            ┌─────────────────────────────────────────────┐
            │   Data layer — src/data/loader.py           │
            │   reads dataset/ecommerce_sessions.csv     │
            │   + FeatureEngineer + PreprocessingPipeline │
            └─────────────────────────────────────────────┘
            ┌─────────────────────────────────────────────┐
            │   MLOps layer                                │
            │   mlruns/  (MLflow UI on :5000)             │
            │   .github/workflows/ci.yml  (lint+test+build)│
            │   docker/ + docker-compose.yml              │
            │   aws/  (ECR + ECS Fargate)                 │
            └─────────────────────────────────────────────┘
```

## Reproducibility chain

| Concern | Mechanism |
|---|---|
| Randomness | `random_seed: 42` in `src/config/config.yaml`, used everywhere (split, CV, model constructors) |
| Library drift | Pinned versions in `requirements.txt` |
| Pipeline drift | The same `models/preprocessing_pipeline.joblib` is loaded at API startup — never refits at serving time |
| Categorical leakage safety | `OneHotEncoder(handle_unknown="ignore")` to absorb unseen cardinalities at inference |
| Python version drift | `pyproject.toml` declares `target-version = ["py310", "py311", "py312"]` |
