from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.backend.interpretation import build_clinical_interpretation
from src.backend.schemas import (
    ExplainResponse,
    ExplainSummaryResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.common import (
    CLASS_DISTRIBUTION_PLOT_PATH,
    CONFUSION_MATRIX_PLOT_PATH,
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE_PLOT_PATH,
    GLOBAL_EXPLANATION_PATH,
    LOCAL_EXPLANATION_EXAMPLE_PATH,
    MODEL_PATH,
    ROC_CURVE_PLOT_PATH,
    SHAP_BAR_PLOT_PATH,
    SHAP_SUMMARY_PLOT_PATH,
    TRAINING_REPORT_PATH,
)
from src.ml.explainability import create_local_explanation


def load_json_artifact(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def build_plot_manifest() -> dict[str, str]:
    return {
        "class_distribution": str(CLASS_DISTRIBUTION_PLOT_PATH),
        "confusion_matrix": str(CONFUSION_MATRIX_PLOT_PATH),
        "roc_curve": str(ROC_CURVE_PLOT_PATH),
        "feature_importance": str(FEATURE_IMPORTANCE_PLOT_PATH),
        "shap_summary": str(SHAP_SUMMARY_PLOT_PATH),
        "shap_bar": str(SHAP_BAR_PLOT_PATH),
    }


@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run 'python main.py' before starting the backend."
        )
    if not TRAINING_REPORT_PATH.exists():
        raise FileNotFoundError(
            "Training report not found. Run 'python main.py' before starting the backend."
        )

    pipeline = joblib.load(MODEL_PATH)
    training_report = json.loads(TRAINING_REPORT_PATH.read_text())
    global_explanation = load_json_artifact(GLOBAL_EXPLANATION_PATH, default={"feature_importance": [], "plot_files": {}})
    example_local_explanation = load_json_artifact(LOCAL_EXPLANATION_EXAMPLE_PATH, default={})

    return {
        "pipeline": pipeline,
        "training_report": training_report,
        "global_explanation": global_explanation,
        "example_local_explanation": example_local_explanation,
        "plot_manifest": build_plot_manifest(),
    }


def build_feature_frame(request: PredictionRequest) -> pd.DataFrame:
    payload = request.model_dump()
    row = {feature: payload[feature] for feature in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def build_prediction_output(feature_frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]
    training_report = bundle["training_report"]

    prediction = int(pipeline.predict(feature_frame)[0])
    probability = 0.0
    if hasattr(pipeline, "predict_proba"):
        probability = float(pipeline.predict_proba(feature_frame)[0][1])

    return (
        {
            "prediction": prediction,
            "label": "Diabetic" if prediction == 1 else "Not Diabetic",
            "probability_diabetic": round(probability, 4),
            "selected_model": training_report["selected_model"],
            "selection_metric": training_report["selection_metric"],
        },
        feature_frame,
    )


def predict_diabetes(request: PredictionRequest) -> PredictionResponse:
    prediction_payload, _ = build_prediction_output(build_feature_frame(request))
    return PredictionResponse(**prediction_payload)


def explain_prediction(request: PredictionRequest) -> ExplainResponse:
    bundle = load_model_bundle()
    feature_frame = build_feature_frame(request)
    prediction_payload, _ = build_prediction_output(feature_frame)

    local_explanation = create_local_explanation(
        bundle["pipeline"],
        feature_frame,
        prediction_label=prediction_payload["label"],
        probability_diabetic=float(prediction_payload["probability_diabetic"]),
    )
    clinical_interpretation = build_clinical_interpretation(
        request.model_dump(),
        prediction_payload["prediction"],
        float(prediction_payload["probability_diabetic"]),
        local_explanation["top_positive_features"],
        local_explanation["top_negative_features"],
    )

    return ExplainResponse(
        **prediction_payload,
        explanation_text=local_explanation["explanation_text"],
        top_positive_features=local_explanation["top_positive_features"],
        top_negative_features=local_explanation["top_negative_features"],
        local_feature_impacts=local_explanation["local_feature_impacts"],
        global_feature_importance=bundle["global_explanation"].get("feature_importance", [])[:10],
        clinical_interpretation=clinical_interpretation,
    )


def get_explain_summary() -> ExplainSummaryResponse:
    bundle = load_model_bundle()
    training_report = bundle["training_report"]
    global_explanation = bundle["global_explanation"]

    return ExplainSummaryResponse(
        selected_model=training_report["selected_model"],
        dataset_used=training_report["dataset_used"],
        selection_metric=training_report["selection_metric"],
        global_feature_importance=global_explanation.get("feature_importance", [])[:12],
        plot_files=bundle["plot_manifest"],
        example_local_explanation=bundle["example_local_explanation"],
    )


def get_metadata() -> MetadataResponse:
    training_report = load_model_bundle()["training_report"]
    return MetadataResponse(
        selected_model=training_report["selected_model"],
        selection_metric=training_report["selection_metric"],
        dataset_used=training_report["dataset_used"],
        feature_columns=training_report["feature_columns"],
    )
