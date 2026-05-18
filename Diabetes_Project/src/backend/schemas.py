from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    HighBP: int = Field(..., ge=0, le=1)
    HighChol: int = Field(..., ge=0, le=1)
    CholCheck: int = Field(..., ge=0, le=1)
    BMI: float = Field(..., ge=12.0, le=98.0)
    Smoker: int = Field(..., ge=0, le=1)
    Stroke: int = Field(..., ge=0, le=1)
    HeartDiseaseorAttack: int = Field(..., ge=0, le=1)
    PhysActivity: int = Field(..., ge=0, le=1)
    Fruits: int = Field(..., ge=0, le=1)
    Veggies: int = Field(..., ge=0, le=1)
    HvyAlcoholConsump: int = Field(..., ge=0, le=1)
    AnyHealthcare: int = Field(..., ge=0, le=1)
    NoDocbcCost: int = Field(..., ge=0, le=1)
    GenHlth: int = Field(..., ge=1, le=5)
    MentHlth: int = Field(..., ge=0, le=30)
    PhysHlth: int = Field(..., ge=0, le=30)
    DiffWalk: int = Field(..., ge=0, le=1)
    Sex: int = Field(..., ge=0, le=1)
    Age: int = Field(..., ge=1, le=13)
    Education: int = Field(..., ge=1, le=6)
    Income: int = Field(..., ge=1, le=8)


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability_diabetic: float
    selected_model: str
    selection_metric: str


class FeatureImpact(BaseModel):
    feature: str
    value: float
    shap_value: float
    absolute_impact: float
    direction: str


class GlobalFeatureImportanceItem(BaseModel):
    feature: str
    mean_abs_shap: float
    mean_shap: float


class ClinicalRule(BaseModel):
    code: str
    effect: str
    message: str


class ClinicalInterpretation(BaseModel):
    risk_level: str
    summary: str
    triggered_rules: list[ClinicalRule]


class ExplainResponse(BaseModel):
    prediction: int
    label: str
    probability_diabetic: float
    selected_model: str
    selection_metric: str
    explanation_text: str
    top_positive_features: list[FeatureImpact]
    top_negative_features: list[FeatureImpact]
    local_feature_impacts: list[FeatureImpact]
    global_feature_importance: list[GlobalFeatureImportanceItem]
    clinical_interpretation: ClinicalInterpretation


class ExplainSummaryResponse(BaseModel):
    selected_model: str
    dataset_used: str
    selection_metric: str
    global_feature_importance: list[GlobalFeatureImportanceItem]
    plot_files: dict[str, str]
    example_local_explanation: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetadataResponse(BaseModel):
    selected_model: str
    selection_metric: str
    dataset_used: str
    feature_columns: list[str]
