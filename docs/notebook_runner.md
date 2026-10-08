# Notebooks

| Notebook | Goal | Run order |
|---|---|---|
| `01_eda.ipynb` | conversion rate, seasonality, correlation heat-map, PageValues leakage signal | 1st |
| `02_features.ipynb` | the five behavioural features + rationale; cross-checks via SHAP + permutation on the re-trained artefacts | 2nd |
| `03_modeling.ipynb` | side-by-side six-model table; CV PR-AUC; value-based threshold | 3rd |
| `04_xai.ipynb` | global SHAP + LIME HTML re-rendered; uses artefacts from `train.bat` | 4th |

## How to run

```bat
call .venv\Scripts\activate.bat
pip install notebook ipykernel   :: Jupyter is not in requirements.txt (install it separately)
jupyter notebook notebooks\01_eda.ipynb
```

Each notebook `sys.path.insert`s the project root so the same `src/` code drives training, the API, the dashboard and the notebooks.
