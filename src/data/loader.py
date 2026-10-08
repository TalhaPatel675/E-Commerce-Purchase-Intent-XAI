"""Typed data loader (PRD §4.1)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.logger import get_logger

log = get_logger(__name__)


class DataValidationError(Exception):
    """Raised when the dataset is malformed (PRD §4.1 "clear error")."""


class DataLoader:
    """Single loader shared by notebooks, API, dashboard (PRD §2.1)."""

    REQUIRED_TARGET = "Converted"

    def __init__(self, csv_path: Path, target_column: str, feature_columns: list[str]) -> None:
        self.csv_path = Path(csv_path)
        self.target_column = target_column
        self.feature_columns = list(feature_columns)

    def load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise DataValidationError(f"Dataset not found at {self.csv_path}")
        log.info("Loading dataset from %s", self.csv_path)
        df = pd.read_csv(self.csv_path)
        log.info("Loaded %d rows × %d columns", len(df), df.shape[1])
        self._validate(df)
        return df

    def split_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise DataValidationError(f"Missing required feature columns: {missing}")
        if self.target_column not in df.columns:
            raise DataValidationError(f"Missing target column: {self.target_column}")
        X = df[self.feature_columns].copy()
        y = df[self.target_column].astype(int).copy()
        return X, y

    def _validate(self, df: pd.DataFrame) -> None:
        if self.REQUIRED_TARGET not in df.columns:
            raise DataValidationError(f"Target column {self.REQUIRED_TARGET!r} missing")
        unique = sorted(df[self.REQUIRED_TARGET].dropna().unique().tolist())
        if not set(unique).issubset({0, 1}):
            raise DataValidationError(f"Target must be 0/1, got {unique}")
        if df.isna().any().any():
            nan_cols = df.columns[df.isna().any()].tolist()
            raise DataValidationError(f"Dataset has NaN values in columns: {nan_cols}")


def load_data_dict(path: Path) -> dict[str, dict[str, str]]:
    df = pd.read_csv(path)
    return {
        row["column"]: {"type": row["type"], "description": row["description"]}
        for _, row in df.iterrows()
    }
