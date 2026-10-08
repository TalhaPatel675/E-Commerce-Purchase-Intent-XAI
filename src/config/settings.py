"""Typed config loader (PRD §4.1 single source of truth)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    dataset_csv: str
    data_dictionary_csv: str
    model_dir: str
    pipeline_path: str
    model_path: str
    threshold_path: str
    metadata_path: str
    reports_dir: str
    figures_dir: str
    mlflow_dir: str
    mlflow_db: str
    ablation_dir: str

    def absolute(self, base: Path) -> PathsConfig:
        kw: dict[str, str] = {}
        for k, v in self.__dict__.items():
            kw[k] = str((base / v).resolve()) if not Path(v).is_absolute() else v
        return PathsConfig(**kw)


@dataclass
class DataConfig:
    target_column: str
    id_columns: list[str]
    categorical_columns: list[str]
    boolean_columns: list[str]
    numeric_columns: list[str]
    test_size: float
    random_seed: int


@dataclass
class PreprocessingConfig:
    imbalance_strategy: str
    scale_numeric: str
    encode_categoricals: str
    unknown_category_handling: str


@dataclass
class ImportanceConfig:
    importance_lenses: list[str]


@dataclass
class FeaturesConfig:
    engineered_features: list[str]
    selection_method: str
    importance: ImportanceConfig = field(
        default_factory=lambda: ImportanceConfig(["shap", "permutation"])
    )


@dataclass
class ValueBasedThresholdConfig:
    intervention_cost: float
    purchase_value: float
    min_recall: float


@dataclass
class PageValuesAblationConfig:
    enabled: bool


@dataclass
class ModellingConfig:
    cv_folds: int
    primary_metric: str
    also_report: list[str]
    n_jobs: int
    value_based_threshold: ValueBasedThresholdConfig
    pagevalues_ablation: PageValuesAblationConfig


@dataclass
class ApiConfig:
    host: str
    port: int
    top_k_contributors: int
    low_confidence_threshold: float
    high_confidence_threshold: float


@dataclass
class DashboardConfig:
    port: int
    api_url: str


@dataclass
class MlflowConfig:
    experiment: str
    registered_model_name: str


@dataclass
class AppConfig:
    paths: PathsConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    features: FeaturesConfig
    modelling: ModellingConfig
    api: ApiConfig
    dashboard: DashboardConfig
    mlflow: MlflowConfig

    @classmethod
    def from_yaml(cls, path: Path) -> AppConfig:
        with open(path, encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        # Nested dataclasses: pop value_based_threshold and pagevalues_ablation out of modelling
        vbt_raw = raw["modelling"].pop("value_based_threshold")
        pva_raw = raw["modelling"].pop("pagevalues_ablation")

        # importance may be a nested map OR a flat list (robust to either)
        imp_section = raw.get("features", {}).pop("importance", None)
        if isinstance(imp_section, dict):
            imp_raw = imp_section.get("lenses", ["shap", "permutation"])
        elif isinstance(imp_section, list):
            imp_raw = imp_section
        else:
            imp_raw = ["shap", "permutation"]

        # Ensure required feature fields present with sensible defaults
        feats_raw = dict(raw["features"])
        feats_raw.setdefault("selection_method", "model_importance")

        paths = PathsConfig(**raw["paths"])
        data = DataConfig(**raw["data"])
        preprocessing = PreprocessingConfig(**raw["preprocessing"])
        features = FeaturesConfig(**feats_raw, importance=ImportanceConfig(imp_raw))
        modelling = ModellingConfig(
            **raw["modelling"],
            value_based_threshold=ValueBasedThresholdConfig(**vbt_raw),
            pagevalues_ablation=PageValuesAblationConfig(**pva_raw),
        )
        api = ApiConfig(**raw["api"])
        dashboard = DashboardConfig(**raw["dashboard"])
        mlflow = MlflowConfig(**raw["mlflow"])

        return cls(
            paths=paths,
            data=data,
            preprocessing=preprocessing,
            features=features,
            modelling=modelling,
            api=api,
            dashboard=dashboard,
            mlflow=mlflow,
        )


def resolve_config(base_dir: Path | None = None) -> AppConfig:
    base = Path(base_dir) if base_dir else Path.cwd()
    cfg_path = base / "src" / "config" / "config.yaml"
    if not cfg_path.exists():
        cfg_path = base / "config.yaml"
    cfg = AppConfig.from_yaml(cfg_path)
    cfg.paths = cfg.paths.absolute(base)
    return cfg
