from fastapi.testclient import TestClient

from src.backend.app import app
from src.common import MODEL_PATH


def test_health_endpoint_returns_status() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    assert "model_loaded" in payload


def test_predict_endpoint_returns_prediction_when_model_exists() -> None:
    if not MODEL_PATH.exists():
        return

    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "HighBP": 1,
            "HighChol": 1,
            "CholCheck": 1,
            "BMI": 31.0,
            "Smoker": 0,
            "Stroke": 0,
            "HeartDiseaseorAttack": 0,
            "PhysActivity": 1,
            "Fruits": 1,
            "Veggies": 1,
            "HvyAlcoholConsump": 0,
            "AnyHealthcare": 1,
            "NoDocbcCost": 0,
            "GenHlth": 3,
            "MentHlth": 2,
            "PhysHlth": 2,
            "DiffWalk": 0,
            "Sex": 0,
            "Age": 8,
            "Education": 5,
            "Income": 6,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] in (0, 1)
    assert payload["label"] in ("Diabetic", "Not Diabetic")
    assert "probability_diabetic" in payload


def test_explain_endpoint_returns_explanation_when_model_exists() -> None:
    if not MODEL_PATH.exists():
        return

    client = TestClient(app)
    response = client.post(
        "/explain",
        json={
            "HighBP": 1,
            "HighChol": 1,
            "CholCheck": 1,
            "BMI": 31.0,
            "Smoker": 0,
            "Stroke": 0,
            "HeartDiseaseorAttack": 0,
            "PhysActivity": 1,
            "Fruits": 1,
            "Veggies": 1,
            "HvyAlcoholConsump": 0,
            "AnyHealthcare": 1,
            "NoDocbcCost": 0,
            "GenHlth": 3,
            "MentHlth": 2,
            "PhysHlth": 2,
            "DiffWalk": 0,
            "Sex": 0,
            "Age": 8,
            "Education": 5,
            "Income": 6,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "top_positive_features" in payload
    assert "top_negative_features" in payload
    assert "clinical_interpretation" in payload
    assert "global_feature_importance" in payload


def test_explain_summary_endpoint_returns_summary_when_model_exists() -> None:
    if not MODEL_PATH.exists():
        return

    client = TestClient(app)
    response = client.get("/explain/summary")

    assert response.status_code == 200
    payload = response.json()
    assert "global_feature_importance" in payload
    assert "plot_files" in payload
