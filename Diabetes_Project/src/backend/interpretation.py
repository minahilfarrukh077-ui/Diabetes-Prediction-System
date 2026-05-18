from __future__ import annotations

from typing import Any


def build_clinical_interpretation(
    payload: dict[str, float],
    prediction: int,
    probability_diabetic: float,
    top_positive_features: list[dict[str, Any]],
    top_negative_features: list[dict[str, Any]],
) -> dict[str, Any]:
    triggered_rules: list[dict[str, str]] = []

    if payload["HighBP"] == 1 and payload["BMI"] >= 30 and payload["GenHlth"] >= 4:
        triggered_rules.append(
            {
                "code": "high_bp_high_bmi_poor_general_health",
                "effect": "increase_risk",
                "message": "High blood pressure, obesity, and poor general health form a high-risk diabetes pattern.",
            }
        )

    if payload["HeartDiseaseorAttack"] == 1 or payload["Stroke"] == 1:
        triggered_rules.append(
            {
                "code": "cardiovascular_history",
                "effect": "increase_risk",
                "message": "Cardiovascular disease history is associated with elevated diabetes risk.",
            }
        )

    if payload["PhysActivity"] == 1 and payload["BMI"] < 25 and payload["HighBP"] == 0:
        triggered_rules.append(
            {
                "code": "active_with_normal_bmi",
                "effect": "decrease_risk",
                "message": "Reported physical activity with a non-obese BMI is a protective pattern.",
            }
        )

    if payload["Fruits"] == 1 and payload["Veggies"] == 1 and payload["HvyAlcoholConsump"] == 0:
        triggered_rules.append(
            {
                "code": "healthier_diet_pattern",
                "effect": "decrease_risk",
                "message": "Fruit and vegetable intake without heavy alcohol consumption supports lower risk.",
            }
        )

    if payload["DiffWalk"] == 1 and payload["GenHlth"] >= 4:
        triggered_rules.append(
            {
                "code": "mobility_and_poor_health",
                "effect": "increase_risk",
                "message": "Difficulty walking together with poor general health suggests a higher-risk profile.",
            }
        )

    if probability_diabetic >= 0.75:
        risk_level = "High"
    elif probability_diabetic >= 0.55:
        risk_level = "Moderate"
    else:
        risk_level = "Lower"

    positive_names = ", ".join(str(item["feature"]) for item in top_positive_features[:3]) or "no strong positive contributors"
    negative_names = ", ".join(str(item["feature"]) for item in top_negative_features[:3]) or "no strong protective contributors"

    summary = (
        f"Risk level is {risk_level.lower()} based on the model probability ({probability_diabetic:.2%}). "
        f"The strongest risk-raising features were {positive_names}, while {negative_names} reduced the score."
    )

    if prediction == 1 and not triggered_rules:
        triggered_rules.append(
            {
                "code": "model_high_risk_without_simple_rule",
                "effect": "increase_risk",
                "message": "The machine learning model detected a higher-risk pattern even though no single rule fired strongly.",
            }
        )

    return {
        "risk_level": risk_level,
        "summary": summary,
        "triggered_rules": triggered_rules,
    }
