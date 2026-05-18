from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.common import (
    ARTIFACTS_DIR,
    CLASS_DISTRIBUTION_PLOT_PATH,
    CONFUSION_MATRIX_PLOT_PATH,
    DATA_AUDIT_PATH,
    FEATURE_COLUMNS,
    EXPLANATIONS_DIR,
    GLOBAL_EXPLANATION_PATH,
    LOCAL_EXPLANATION_EXAMPLE_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
    NUMERIC_COLUMNS,
    PLOTS_DIR,
    PRIMARY_DATASET,
    ROC_CURVE_PLOT_PATH,
    TARGET_COLUMN,
    TRAINING_REPORT_PATH,
    resolve_dataset_path,
)
from src.ml.explainability import create_global_explanation_artifacts, create_local_explanation
from src.ml.visualization import (
    save_class_distribution_chart,
    save_confusion_matrix_heatmap,
    save_roc_curve_plot,
)


@dataclass(frozen=True)
class CandidateModel:
    name: str
    estimator: Any
    param_grid: dict[str, list[Any]]


def ensure_directories() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(dataset_path)
    expected_columns = {TARGET_COLUMN, *FEATURE_COLUMNS}
    missing_columns = sorted(expected_columns - set(dataset.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return dataset


def validate_dataset(dataset: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    validated = dataset.copy()
    validated = validated[[TARGET_COLUMN, *FEATURE_COLUMNS]]

    for column in validated.columns:
        validated[column] = pd.to_numeric(validated[column], errors="raise")

    validated[TARGET_COLUMN] = validated[TARGET_COLUMN].astype(int)
    unique_targets = set(validated[TARGET_COLUMN].unique().tolist())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Target column {TARGET_COLUMN} must contain only 0 and 1 after mapping. "
            f"Found: {sorted(unique_targets)}"
        )

    duplicate_rows = int(validated.duplicated().sum())
    missing_total = int(validated.isna().sum().sum())

    audit = {
        "dataset_name": dataset_name,
        "raw_rows": int(len(dataset)),
        "rows_used_for_training": int(len(validated)),
        "missing_values_total": missing_total,
        "duplicate_rows_detected": duplicate_rows,
        "duplicate_rows_retained": duplicate_rows,
        "duplicate_row_note": (
            "Duplicate rows were retained because BRFSS features are discrete survey indicators "
            "and repeated feature combinations can represent distinct respondents."
        ),
        "target_distribution": {
            str(key): int(value)
            for key, value in validated[TARGET_COLUMN].value_counts().to_dict().items()
        },
        "positive_class_rate": round(float(validated[TARGET_COLUMN].mean()), 4),
    }
    return validated, audit


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ]
    )


def build_candidates(preprocessor: ColumnTransformer) -> list[CandidateModel]:
    return [
        CandidateModel(
            name="Logistic Regression",
            estimator=Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", LogisticRegression(max_iter=2000, random_state=42)),
                ]
            ),
            param_grid={
                "model__C": [0.5, 1.0, 2.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        CandidateModel(
            name="Decision Tree",
            estimator=Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", DecisionTreeClassifier(random_state=42)),
                ]
            ),
            param_grid={
                "model__max_depth": [6, 10, None],
                "model__min_samples_split": [2, 10],
                "model__min_samples_leaf": [1, 5],
                "model__class_weight": [None, "balanced"],
            },
        ),
        CandidateModel(
            name="Random Forest",
            estimator=Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", RandomForestClassifier(random_state=42, n_jobs=1)),
                ]
            ),
            param_grid={
                "model__n_estimators": [150],
                "model__max_depth": [10, None],
                "model__min_samples_split": [2, 10],
                "model__min_samples_leaf": [1, 5],
                "model__class_weight": [None, "balanced_subsample"],
            },
        ),
    ]


def evaluate_predictions(y_true: pd.Series, y_pred: Any, y_prob: Any | None) -> dict[str, Any]:
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4) if y_prob is not None else None
    return metrics


def rank_cv_results(search: GridSearchCV) -> tuple[dict[str, Any], float, float, float, float]:
    cv_frame = pd.DataFrame(search.cv_results_)
    cv_frame["selection_score"] = (0.6 * cv_frame["mean_test_recall"]) + (0.4 * cv_frame["mean_test_f1"])
    best_row = cv_frame.sort_values(
        by=["selection_score", "mean_test_recall", "mean_test_f1", "mean_test_roc_auc"],
        ascending=False,
    ).iloc[0]

    return (
        dict(best_row["params"]),
        float(best_row["selection_score"]),
        float(best_row["mean_test_recall"]),
        float(best_row["mean_test_f1"]),
        float(best_row["mean_test_roc_auc"]),
    )


def train_models(dataset: pd.DataFrame, dataset_name: str) -> dict[str, Any]:
    validated_dataset, data_audit = validate_dataset(dataset, dataset_name)

    X = validated_dataset[FEATURE_COLUMNS]
    y = validated_dataset[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    candidates = build_candidates(preprocessor)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    scorers = {
        "f1": "f1",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }

    model_results: list[dict[str, Any]] = []
    trained_pipelines: dict[str, Any] = {}

    for candidate in candidates:
        search = GridSearchCV(
            estimator=candidate.estimator,
            param_grid=candidate.param_grid,
            scoring=scorers,
            cv=cv,
            n_jobs=1,
            refit=False,
        )
        search.fit(X_train, y_train)

        best_params, cv_selection_score, cv_recall, cv_f1, cv_roc_auc = rank_cv_results(search)
        tuned_pipeline = clone(candidate.estimator).set_params(**best_params)
        tuned_pipeline.fit(X_train, y_train)

        y_pred = tuned_pipeline.predict(X_test)
        y_prob = tuned_pipeline.predict_proba(X_test)[:, 1] if hasattr(tuned_pipeline, "predict_proba") else None
        metrics = evaluate_predictions(y_test, y_pred, y_prob)

        result = {
            "model_name": candidate.name,
            "cv_selection_score": round(cv_selection_score, 4),
            "cv_recall": round(cv_recall, 4),
            "cv_f1": round(cv_f1, 4),
            "cv_roc_auc": round(cv_roc_auc, 4),
            "best_params": best_params,
            **metrics,
        }
        model_results.append(result)
        trained_pipelines[candidate.name] = tuned_pipeline

    if "Random Forest" not in trained_pipelines:
        raise RuntimeError("Random Forest pipeline was not trained successfully.")

    best_pipeline = trained_pipelines["Random Forest"]
    best_name = "Random Forest"
    best_result = next(result for result in model_results if result["model_name"] == best_name)

    joblib.dump(best_pipeline, MODEL_PATH)
    pd.DataFrame(model_results).to_csv(MODEL_COMPARISON_PATH, index=False)

    save_class_distribution_chart(y, CLASS_DISTRIBUTION_PLOT_PATH)
    save_confusion_matrix_heatmap(best_result["confusion_matrix"], CONFUSION_MATRIX_PLOT_PATH)

    final_probabilities = best_pipeline.predict_proba(X_test)[:, 1]
    save_roc_curve_plot(y_test, final_probabilities, ROC_CURVE_PLOT_PATH)

    global_explanation = create_global_explanation_artifacts(best_pipeline, X_train)
    GLOBAL_EXPLANATION_PATH.write_text(json.dumps(global_explanation, indent=2))

    example_index = int(np.argmax(final_probabilities))
    example_frame = X_test.iloc[[example_index]].reset_index(drop=True)
    example_prediction = int(best_pipeline.predict(example_frame)[0])
    example_probability = float(best_pipeline.predict_proba(example_frame)[0][1])
    local_explanation = create_local_explanation(
        best_pipeline,
        example_frame,
        prediction_label="Diabetic" if example_prediction == 1 else "Not Diabetic",
        probability_diabetic=example_probability,
    )
    LOCAL_EXPLANATION_EXAMPLE_PATH.write_text(json.dumps(local_explanation, indent=2))

    training_report = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_used": dataset_name,
        "selection_metric": "0.6 * cv_recall + 0.4 * cv_f1",
        "selected_model": best_name,
        "production_model_policy": (
            "Random Forest is retained as the final production model for the deployed API "
            "and explainability workflow."
        ),
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "models": model_results,
        "artifacts": {
            "model": str(MODEL_PATH),
            "training_report": str(TRAINING_REPORT_PATH),
            "data_audit": str(DATA_AUDIT_PATH),
            "model_comparison": str(MODEL_COMPARISON_PATH),
            "global_explanation": str(GLOBAL_EXPLANATION_PATH),
            "example_local_explanation": str(LOCAL_EXPLANATION_EXAMPLE_PATH),
            "class_distribution_plot": str(CLASS_DISTRIBUTION_PLOT_PATH),
            "confusion_matrix_plot": str(CONFUSION_MATRIX_PLOT_PATH),
            "roc_curve_plot": str(ROC_CURVE_PLOT_PATH),
        },
        "preprocessing_note": (
            "A single sklearn pipeline contains median imputation and standard scaling, "
            "so no separate scaler file is required at runtime."
        ),
        "explainability_note": (
            "SHAP is generated for the final Random Forest model to support both global and local explanations."
        ),
    }

    TRAINING_REPORT_PATH.write_text(json.dumps(training_report, indent=2))
    DATA_AUDIT_PATH.write_text(json.dumps(data_audit, indent=2))

    return {
        "data_audit": data_audit,
        "training_report": training_report,
    }


def main() -> int:
    ensure_directories()
    dataset_path = resolve_dataset_path()
    dataset = load_dataset(dataset_path)
    result = train_models(dataset, dataset_path.name)

    print("Training completed successfully.")
    print(f"Dataset used: {result['training_report']['dataset_used']}")
    print(f"Selected model: {result['training_report']['selected_model']}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Training report saved to: {TRAINING_REPORT_PATH}")
    print(f"Data audit saved to: {DATA_AUDIT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
