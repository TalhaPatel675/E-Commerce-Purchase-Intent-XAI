# Evaluation-rubric mapping (PRD §10)

| Bucket | Weight | Evidence |
|---|---|---|
| Data prep & EDA | 15% | ColumnTransformer in `src/preprocessing/preprocess.py`; categorical handling including int IDs; EDA notebook 01 with correlated heat-map; seasonality; class-balance; saved pipeline |
| Modelling & evaluation | 20% | Six algorithms through one trainer (`src/models/trainer.py`); stratified 5-fold CV; PR-AUC-led metrics; value-based threshold optimisation; PageValues ablation; artefact persistence |
| Explainability (XAI) | 15% | SHAP global + LIME local + permutation importance (lens #2) + model card; surfaced in API (`/predict` returns top contributors); dashboard Explainability section |
| API & dashboard | 20% | Pydantic FastAPI service with `/predict`, `/health`, `/docs`, load-once artefacts; six-section Streamlit with Plotly; live scoring form posts to the API and renders contributors |
| Deployment & MLOps | 15% | `docker-compose.yml` (api + dashboard + MLflow + trainer); Dockerfiles lean + non-root + libgomp1 + `.dockerignore`; MLflow experiment + registered model version (created by `scripts\train.bat`; the ZIP ships an empty `mlruns/`); GitHub Actions matrix CI + Docker build (workflow included, never run on GitHub); AWS ECR/ECS notes |
| Code quality | 10% | OOP trainers/loaders/pipelines; type hints; `logging` module; explicit exceptions; black + ruff; pytest (9 tests); deterministic seeds |
| Documentation | 5% | README; install (Windows); user guide; architecture diagram; PageValues ablation; notebook runner; rubric map; requirements trace |
| Demo recording (§9) | required | 3–5 min screen capture of `/predict` + dashboard — produced by you on Windows |
