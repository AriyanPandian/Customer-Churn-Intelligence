"""Train and evaluate the telecom customer churn prediction pipeline."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline.pkl"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "Contract",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "PaymentMethod",
    "PaperlessBilling",
    "SeniorCitizen",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_dataset() -> pd.DataFrame:
    """Download the source dataset once and return the cached CSV."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print(f"Downloading customer churn dataset to {DATA_PATH.relative_to(ROOT)}...")
        urllib.request.urlretrieve(DATASET_URL, DATA_PATH)

    frame = pd.read_csv(DATA_PATH)
    required = set(FEATURES + ["Churn"])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return frame


def prepare_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean selected fields and return model features and a binary target."""
    data = frame[FEATURES + ["Churn"]].copy()
    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    imputed_total = data["MonthlyCharges"] * data["tenure"]
    imputed_total = imputed_total.where(data["tenure"] != 0, 0)
    data["TotalCharges"] = data["TotalCharges"].fillna(imputed_total)
    data["SeniorCitizen"] = data["SeniorCitizen"].map({1: "Yes", 0: "No"}).fillna(
        data["SeniorCitizen"].astype(str)
    )

    target = data.pop("Churn").map({"Yes": 1, "No": 0})
    if target.isna().any():
        raise ValueError("Churn target contains values other than 'Yes' or 'No'.")
    return data[FEATURES], target.astype(int)


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    """Build the preprocessing and XGBoost estimator as one inference pipeline."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    classifier = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def collect_feature_importances(pipeline: Pipeline) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return importances by original feature and by transformed feature."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    encoded: list[dict[str, Any]] = []
    grouped: dict[str, float] = {feature: 0.0 for feature in FEATURES}

    for name, importance in zip(names, classifier.feature_importances_, strict=True):
        display_name = str(name).split("__", 1)[-1]
        source_feature = next(
            feature
            for feature in FEATURES
            if display_name == feature or display_name.startswith(f"{feature}_")
        )
        value = float(importance)
        grouped[source_feature] += value
        encoded.append({"feature": display_name, "importance": value})

    encoded.sort(key=lambda item: item["importance"], reverse=True)
    aggregated = [
        {"feature": feature, "importance": importance}
        for feature, importance in grouped.items()
    ]
    aggregated.sort(key=lambda item: item["importance"], reverse=True)
    return aggregated, encoded[:12]


def train() -> dict[str, Any]:
    """Train, evaluate, serialize the model, and save reproducibility metadata."""
    frame = load_dataset()
    features, target = prepare_data(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)
    pipeline = build_pipeline(negative / positive)
    pipeline.fit(x_train, y_train)

    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, predictions)),
    }
    aggregated, encoded = collect_feature_importances(pipeline)
    metadata: dict[str, Any] = {
        "metrics": metrics,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "top_feature_importances": aggregated[:12],
        "top_encoded_feature_importances": encoded,
        "baseline_churn_rate": float(target.mean()),
        "dataset_rows": int(len(frame)),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("\nHoldout test metrics")
    print("--------------------")
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title():<12} {value:.4f}")
    print(f"{'TN / FP':<12} {tn} / {fp}")
    print(f"{'FN / TP':<12} {fn} / {tp}")
    print(f"\nSaved model: {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved metadata: {METADATA_PATH.relative_to(ROOT)}")
    return metadata


if __name__ == "__main__":
    train()
