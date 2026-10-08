"""End-to-end training pipeline for Purchase Intent XAI."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline

from src.config.settings import resolve_config
from src.data.loader import DataLoader
from src.explain.shap_lime import generate_explainability_artifacts
from src.features.engineer import FeatureEngineer
from src.models.ablation import run_pagevalues_ablation
from src.models.trainer import Trainer, default_model_factory
from src.preprocessing.preprocess import PreprocessingPipeline
from src.utils.logger import configure_logging, get_logger
from src.utils.metrics import full_metric_suite, value_based_threshold

log = get_logger(__name__)


def select_best(results: dict[str, dict[str, Any]], primary: str = "cv_pr_auc_mean") -> str:
    """Select the winner using cross-validation, never the held-out test set."""
    return max(results.items(), key=lambda kv: kv[1].get(primary, -1.0))[0]


def _cv_predictions(model_factory, X_df, y, pipeline_builder, cv):
    """Generate leakage-safe out-of-fold probabilities."""
    y = np.asarray(y)
    oof = np.zeros(len(y), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_df, y), start=1):
        pipe = pipeline_builder()
        X_tr = pipe.fit_transform(X_df.iloc[tr_idx])
        X_va = pipe.transform(X_df.iloc[va_idx])
        model = model_factory()
        model.fit(X_tr, y[tr_idx])
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        log.info("CV fold %d/%d complete", fold, cv.n_splits)
    return oof


def evaluate_models(
    X_train_df,
    X_test_df,
    y_train,
    y_test,
    pipeline_builder,
    seed: int,
    cv_folds: int,
) -> dict[str, dict[str, Any]]:
    """Train six models, select by leakage-safe CV PR-AUC, then evaluate on test."""
    results: dict[str, dict[str, Any]] = {}
    factories = default_model_factory(seed)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    X_train_df = X_train_df.reset_index(drop=True)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    for name, cfg in factories.items():
        log.info("=== %s ===", name)
        t0 = time.time()
        try:
            # Held-out test evaluation uses a preprocessor fitted only on X_train.
            pipe = pipeline_builder()
            X_tr = pipe.fit_transform(X_train_df)
            X_te = pipe.transform(X_test_df)
            model = cfg.factory()
            Trainer(name, model).fit(X_tr, y_train)
            test_proba = model.predict_proba(X_te)[:, 1]
            metrics = full_metric_suite(y_test, (test_proba >= 0.5).astype(int), test_proba)

            # CV preprocessing is fitted independently inside every fold.
            oof_proba = _cv_predictions(cfg.factory, X_train_df, y_train, pipeline_builder, cv)
            fold_scores = [
                average_precision_score(y_train[va_idx], oof_proba[va_idx])
                for _, va_idx in cv.split(X_train_df, y_train)
            ]
            metrics["cv_pr_auc_mean"] = float(np.mean(fold_scores))
            metrics["cv_pr_auc_std"] = float(np.std(fold_scores))
            metrics["fit_seconds"] = float(time.time() - t0)
            results[name] = metrics
            log.info(
                "%s: CV PR-AUC=%.4f ± %.4f | test PR-AUC=%.4f",
                name,
                metrics["cv_pr_auc_mean"],
                metrics["cv_pr_auc_std"],
                metrics["pr_auc"],
            )
        except Exception as exc:
            log.exception("Model %s failed: %s", name, exc)

    return results


def _expected_value(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    purchase_value: float,
    intervention_cost: float,
) -> float:
    pred = (y_proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    return float(tp * purchase_value - fp * intervention_cost)


def main() -> None:
    configure_logging(level="INFO")
    cfg = resolve_config(ROOT)

    # ---------- DATA + FEATURE ENGINEERING ----------
    loader = DataLoader(
        csv_path=Path(cfg.paths.dataset_csv),
        target_column=cfg.data.target_column,
        feature_columns=cfg.data.numeric_columns + cfg.data.categorical_columns,
    )
    df = loader.load()
    X_raw, y = loader.split_xy(df)
    X = FeatureEngineer().fit_transform(X_raw)

    all_numeric = cfg.data.numeric_columns + FeatureEngineer.ENGINEERED
    numeric_columns = [c for c in all_numeric if c in X.columns]
    categorical_columns = [c for c in cfg.data.categorical_columns if c in X.columns]

    def pipeline_builder() -> PreprocessingPipeline:
        return PreprocessingPipeline(
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            scale_numeric=cfg.preprocessing.scale_numeric,
        )

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_seed,
        stratify=y,
    )
    X_train_df = X_train_df.reset_index(drop=True)
    X_test_df = X_test_df.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # ---------- MLFLOW SETUP ----------
    import mlflow

    mlflow_db = Path(cfg.paths.mlflow_db).resolve()
    mlflow_db.parent.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{mlflow_db.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)

    # ---------- MODEL BENCHMARK ----------
    results = evaluate_models(
        X_train_df,
        X_test_df,
        y_train.values,
        y_test.values,
        pipeline_builder,
        cfg.data.random_seed,
        cfg.modelling.cv_folds,
    )
    if not results:
        raise RuntimeError("No model trained successfully")

    best_name = select_best(results, cfg.modelling.primary_metric)
    log.info("Best model by %s: %s", cfg.modelling.primary_metric, best_name)
    factories = default_model_factory(cfg.data.random_seed)
    best_cfg = factories[best_name]

    # Log benchmark metadata to MLflow.
    with mlflow.start_run(run_name="cv-benchmark"):
        mlflow.log_params(
            {
                "primary_metric": cfg.modelling.primary_metric,
                "cv_folds": cfg.modelling.cv_folds,
                "random_seed": cfg.data.random_seed,
            }
        )
        for name, metrics in results.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_param("model", name)
                mlflow.log_metrics(
                    {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
                )

    # ---------- REFIT BEST MODEL ON TRAINING DATA ONLY ----------
    pipeline = pipeline_builder()
    X_train = pipeline.fit_transform(X_train_df)
    X_test = pipeline.transform(X_test_df)
    best_model = best_cfg.factory()
    best_model.fit(X_train, y_train.values)
    test_proba = best_model.predict_proba(X_test)[:, 1]

    # ---------- BUSINESS THRESHOLD: SELECT ON OOF, EVALUATE ON TEST ----------
    cv = StratifiedKFold(
        n_splits=cfg.modelling.cv_folds,
        shuffle=True,
        random_state=cfg.data.random_seed,
    )
    oof_proba = _cv_predictions(best_cfg.factory, X_train_df, y_train.values, pipeline_builder, cv)
    oof_threshold = value_based_threshold(
        y_true=y_train.values,
        y_proba=oof_proba,
        intervention_cost=cfg.modelling.value_based_threshold.intervention_cost,
        purchase_value=cfg.modelling.value_based_threshold.purchase_value,
        min_recall=cfg.modelling.value_based_threshold.min_recall,
    )
    threshold = float(oof_threshold["threshold"])
    test_pred_threshold = (test_proba >= threshold).astype(int)
    test_threshold_metrics = full_metric_suite(y_test.values, test_pred_threshold, test_proba)
    test_expected_value = _expected_value(
        y_test.values,
        test_proba,
        threshold,
        cfg.modelling.value_based_threshold.purchase_value,
        cfg.modelling.value_based_threshold.intervention_cost,
    )
    threshold_info = {
        "threshold": threshold,
        "expected_value": float(test_expected_value),
        "precision": float(test_threshold_metrics["precision"]),
        "recall": float(test_threshold_metrics["recall"]),
        "f1": float(test_threshold_metrics["f1"]),
        "oof_expected_value": float(oof_threshold["expected_value"]),
        "oof_precision": float(oof_threshold["precision"]),
        "oof_recall": float(oof_threshold["recall"]),
    }
    log.info(
        "Selected threshold %.4f from OOF; test precision=%.4f recall=%.4f",
        threshold,
        threshold_info["precision"],
        threshold_info["recall"],
    )

    # ---------- PAGEVALUES ABLATION ----------
    ablation_info: dict[str, Any] = {}
    if cfg.modelling.pagevalues_ablation.enabled:
        ablation_info = run_pagevalues_ablation(
            best_name=best_name,
            best_config=best_cfg,
            pipeline_factory=PreprocessingPipeline,
            X_train_df=X_train_df,
            y_train=y_train.values,
            X_test_df=X_test_df,
            y_test=y_test.values,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            feature_columns_drop=("PageValues", "PageValuePerProduct"),
            cv_folds=cfg.modelling.cv_folds,
            random_seed=cfg.data.random_seed,
        )
        ablation_path = Path(cfg.paths.ablation_dir) / "pagevalues_ablation.json"
        ablation_path.parent.mkdir(parents=True, exist_ok=True)
        ablation_path.write_text(json.dumps(ablation_info, indent=2), encoding="utf-8")

    # ---------- MLFLOW: LOG + REGISTER A COMPLETE RAW-INPUT PIPELINE ----------
    # The registered artifact includes feature engineering + fitted preprocessing + model,
    # so its signature matches the raw session schema rather than the transformed matrix.
    mlflow_model = SklearnPipeline(
        [
            ("feature_engineering", FeatureEngineer()),
            ("preprocessing", pipeline.transformer_),
            ("model", best_model),
        ]
    )
    sample_raw = X_raw.iloc[:3].copy()
    sample_pred = best_model.predict_proba(
        pipeline.transform(FeatureEngineer().fit_transform(sample_raw))
    )[:, 1]

    from mlflow.models.signature import infer_signature

    signature = infer_signature(sample_raw, sample_pred)
    with mlflow.start_run(run_name="BEST-" + best_name):
        mlflow.set_tag("best_model", best_name)
        mlflow.log_params(
            {
                "random_seed": cfg.data.random_seed,
                "primary_metric": cfg.modelling.primary_metric,
                "selection_basis": "leakage_safe_cv_pr_auc",
            }
        )
        mlflow.log_metrics(
            {
                "cv_pr_auc_mean": results[best_name]["cv_pr_auc_mean"],
                "cv_pr_auc_std": results[best_name]["cv_pr_auc_std"],
                "test_pr_auc": results[best_name]["pr_auc"],
                "threshold": threshold,
                "test_expected_value": threshold_info["expected_value"],
                "test_recall_at_threshold": threshold_info["recall"],
                "test_precision_at_threshold": threshold_info["precision"],
            }
        )
        if ablation_info:
            mlflow.log_metrics(
                {
                    "ablation_test_pr_auc_without_PageValues": ablation_info["metrics"]["pr_auc"],
                    "pr_auc_gap_with_vs_without_PageValues": (
                        results[best_name]["pr_auc"] - ablation_info["metrics"]["pr_auc"]
                    ),
                }
            )
        model_info = mlflow.sklearn.log_model(
            sk_model=mlflow_model,
            artifact_path="model",
            signature=signature,
            input_example=sample_raw,
            registered_model_name=cfg.mlflow.registered_model_name,
        )
        log.info("Registered model URI: %s", model_info.model_uri)

    # ---------- PERSIST LOCAL ARTIFACTS ----------
    Path(cfg.paths.model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, cfg.paths.model_path)
    pipeline.save(Path(cfg.paths.pipeline_path))
    Path(cfg.paths.threshold_path).write_text(
        json.dumps(
            {
                "threshold": threshold,
                "selected_model": best_name,
                "expected_value": threshold_info["expected_value"],
                "precision": threshold_info["precision"],
                "recall": threshold_info["recall"],
                "f1": threshold_info["f1"],
                "oof_expected_value": threshold_info["oof_expected_value"],
                "oof_precision": threshold_info["oof_precision"],
                "oof_recall": threshold_info["oof_recall"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    Path(cfg.paths.metadata_path).write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "selection_metric": cfg.modelling.primary_metric,
                "selection_basis": "leakage_safe_cv_pr_auc",
                "all_results": results,
                "threshold_info": threshold_info,
                "engineered_features": FeatureEngineer.ENGINEERED,
                "pagevalues_ablation": ablation_info,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------- EXPLAINABILITY ----------
    full_X = pipeline.transform(X)
    explainability_summary = generate_explainability_artifacts(
        best_model=best_model,
        X_full=full_X,
        y_full=y,
        feature_names=pipeline.get_feature_names(),
        reports_dir=Path(cfg.paths.reports_dir),
        figures_dir=Path(cfg.paths.figures_dir),
        data_dictionary_csv=Path(cfg.paths.data_dictionary_csv),
        best_name=best_name,
        results=results,
        threshold_info=threshold_info,
        ablation_info=ablation_info,
    )
    failed_lenses = [
        name
        for name in ("global_shap", "permutation_importance", "local_lime")
        if not explainability_summary.get(name, {}).get("ok", False)
    ]
    if failed_lenses:
        raise RuntimeError(
            "Required explainability artifact(s) failed: "
            + ", ".join(failed_lenses)
            + ". Install the pinned dependencies and rerun scripts\\train.bat."
        )

    log.info(
        "Training complete. Artifacts dir=%s, mlruns dir=%s",
        cfg.paths.model_dir,
        cfg.paths.mlflow_dir,
    )


if __name__ == "__main__":
    main()
