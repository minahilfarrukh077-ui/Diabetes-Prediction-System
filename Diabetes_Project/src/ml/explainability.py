from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diabetes_project_mpl")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.common import FEATURE_IMPORTANCE_PLOT_PATH, SHAP_BAR_PLOT_PATH, SHAP_SUMMARY_PLOT_PATH
from src.ml.visualization import save_feature_importance_bar_chart


def simplify_feature_names(feature_names: list[str] | np.ndarray) -> list[str]:
    return [str(name).split("__", 1)[-1] for name in feature_names]


def get_pipeline_components(pipeline: Any) -> tuple[Any, Any, list[str]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = simplify_feature_names(preprocessor.get_feature_names_out())
    return preprocessor, model, feature_names


def transform_feature_frame(pipeline: Any, feature_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    preprocessor, _, feature_names = get_pipeline_components(pipeline)
    transformed = preprocessor.transform(feature_frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed_array = np.asarray(transformed, dtype=float)
    transformed_frame = pd.DataFrame(transformed_array, columns=feature_names, index=feature_frame.index)
    return transformed_frame, feature_names


def normalize_binary_shap_output(raw_values: Any, expected_value: Any) -> tuple[np.ndarray, float]:
    expected_array = np.asarray(expected_value)

    if isinstance(raw_values, list):
        class_index = 1 if len(raw_values) > 1 else 0
        shap_matrix = np.asarray(raw_values[class_index], dtype=float)
        if expected_array.ndim == 0:
            base_value = float(expected_array)
        else:
            base_value = float(expected_array[class_index])
        return shap_matrix, base_value

    shap_array = np.asarray(raw_values, dtype=float)
    if shap_array.ndim == 3:
        class_index = 1 if shap_array.shape[-1] > 1 else 0
        shap_matrix = shap_array[:, :, class_index]
        if expected_array.ndim == 0:
            base_value = float(expected_array)
        else:
            base_value = float(np.ravel(expected_array)[class_index])
        return shap_matrix, base_value

    if shap_array.ndim == 2:
        if expected_array.ndim == 0:
            base_value = float(expected_array)
        else:
            base_value = float(np.ravel(expected_array)[0])
        return shap_array, base_value

    raise ValueError("Unsupported SHAP output shape for binary classification.")


def compute_shap_matrix(pipeline: Any, feature_frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, float]:
    transformed_frame, _ = transform_feature_frame(pipeline, feature_frame)
    _, model, _ = get_pipeline_components(pipeline)
    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(transformed_frame)
    shap_matrix, base_value = normalize_binary_shap_output(raw_values, explainer.expected_value)
    return transformed_frame, shap_matrix, base_value


def build_feature_importance_items(feature_names: list[str], shap_matrix: np.ndarray) -> list[dict[str, float | str]]:
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    mean_signed = shap_matrix.mean(axis=0)

    items = [
        {
            "feature": feature_name,
            "mean_abs_shap": round(float(abs_score), 6),
            "mean_shap": round(float(mean_score), 6),
        }
        for feature_name, abs_score, mean_score in zip(feature_names, mean_abs, mean_signed, strict=False)
    ]
    return sorted(items, key=lambda item: float(item["mean_abs_shap"]), reverse=True)


def build_local_feature_impacts(
    feature_frame: pd.DataFrame,
    shap_vector: np.ndarray,
    *,
    top_n: int = 5,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]], list[dict[str, float | str]]]:
    record = feature_frame.iloc[0].to_dict()
    feature_impacts = []

    for feature_name, shap_value in zip(feature_frame.columns.tolist(), shap_vector.tolist(), strict=False):
        feature_impacts.append(
            {
                "feature": feature_name,
                "value": float(record[feature_name]),
                "shap_value": round(float(shap_value), 6),
                "absolute_impact": round(abs(float(shap_value)), 6),
                "direction": "increase_risk" if float(shap_value) >= 0 else "decrease_risk",
            }
        )

    ordered = sorted(feature_impacts, key=lambda item: float(item["absolute_impact"]), reverse=True)
    top_positive = [item for item in ordered if float(item["shap_value"]) > 0][:top_n]
    top_negative = [item for item in ordered if float(item["shap_value"]) < 0][:top_n]

    return ordered, top_positive, top_negative


def summarize_local_explanation(
    prediction_label: str,
    probability_diabetic: float,
    top_positive: list[dict[str, float | str]],
    top_negative: list[dict[str, float | str]],
) -> str:
    positive_names = ", ".join(str(item["feature"]) for item in top_positive[:3]) or "no dominant risk-raising factors"
    negative_names = ", ".join(str(item["feature"]) for item in top_negative[:3]) or "no dominant protective factors"

    return (
        f"The model predicts '{prediction_label}' with a diabetes probability of {probability_diabetic:.2%}. "
        f"The strongest factors pushing risk upward were {positive_names}. "
        f"The strongest factors reducing risk were {negative_names}."
    )


def create_global_explanation_artifacts(
    pipeline: Any,
    feature_frame: pd.DataFrame,
    *,
    summary_plot_path: Path = SHAP_SUMMARY_PLOT_PATH,
    shap_bar_plot_path: Path = SHAP_BAR_PLOT_PATH,
    feature_importance_plot_path: Path = FEATURE_IMPORTANCE_PLOT_PATH,
    sample_size: int = 1200,
) -> dict[str, Any]:
    sample_frame = feature_frame.sample(n=min(sample_size, len(feature_frame)), random_state=42).reset_index(drop=True)
    transformed_frame, shap_matrix, _ = compute_shap_matrix(pipeline, sample_frame)
    feature_items = build_feature_importance_items(transformed_frame.columns.tolist(), shap_matrix)

    save_feature_importance_bar_chart(
        feature_items,
        feature_importance_plot_path,
        title="Global Feature Importance (Mean |SHAP|)",
    )

    summary_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_matrix, transformed_frame, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8.5, 6))
    shap.summary_plot(shap_matrix, transformed_frame, plot_type="bar", max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(shap_bar_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "sample_size": int(len(sample_frame)),
        "feature_importance": feature_items,
        "plot_files": {
            "feature_importance": str(feature_importance_plot_path),
            "shap_summary": str(summary_plot_path),
            "shap_bar": str(shap_bar_plot_path),
        },
    }


def create_local_explanation(
    pipeline: Any,
    feature_frame: pd.DataFrame,
    *,
    prediction_label: str,
    probability_diabetic: float,
    top_n: int = 5,
) -> dict[str, Any]:
    _, shap_matrix, base_value = compute_shap_matrix(pipeline, feature_frame)
    shap_vector = shap_matrix[0]
    local_feature_impacts, top_positive, top_negative = build_local_feature_impacts(
        feature_frame,
        shap_vector,
        top_n=top_n,
    )
    explanation_text = summarize_local_explanation(
        prediction_label,
        probability_diabetic,
        top_positive,
        top_negative,
    )

    return {
        "base_value": round(float(base_value), 6),
        "feature_values": {
            feature: float(value) for feature, value in feature_frame.iloc[0].to_dict().items()
        },
        "local_feature_impacts": local_feature_impacts,
        "top_positive_features": top_positive,
        "top_negative_features": top_negative,
        "explanation_text": explanation_text,
    }
