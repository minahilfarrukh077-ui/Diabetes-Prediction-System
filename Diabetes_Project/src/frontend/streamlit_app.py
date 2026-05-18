from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from src.common import (
    CLASS_DISTRIBUTION_PLOT_PATH,
    CONFUSION_MATRIX_PLOT_PATH,
    DEFAULT_INPUTS,
    FEATURE_IMPORTANCE_PLOT_PATH,
    FIELD_GROUPS,
    FIELD_HELP,
    GLOBAL_EXPLANATION_PATH,
    LOCAL_EXPLANATION_EXAMPLE_PATH,
    OPTION_LABELS,
    ROC_CURVE_PLOT_PATH,
    SHAP_BAR_PLOT_PATH,
    SHAP_SUMMARY_PLOT_PATH,
    TRAINING_REPORT_PATH,
    VALIDATION_BOUNDS,
)

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
PAGE_OPTIONS = ["Prediction", "Model Dashboard", "Explainability"]

DISPLAY_LABELS = {
    "HighBP": "High Blood Pressure",
    "HighChol": "High Cholesterol",
    "CholCheck": "Cholesterol Check (Last 5 Years)",
    "BMI": "Body Mass Index (BMI)",
    "Smoker": "Smoking History",
    "Stroke": "Stroke History",
    "HeartDiseaseorAttack": "Heart Disease / Heart Attack History",
    "PhysActivity": "Physical Activity",
    "Fruits": "Fruit Intake",
    "Veggies": "Vegetable Intake",
    "HvyAlcoholConsump": "Heavy Alcohol Consumption",
    "AnyHealthcare": "Healthcare Coverage",
    "NoDocbcCost": "Skipped Doctor Due To Cost",
    "GenHlth": "General Health",
    "MentHlth": "Poor Mental Health Days",
    "PhysHlth": "Poor Physical Health Days",
    "DiffWalk": "Difficulty Walking",
    "Sex": "Sex",
    "Age": "Age Group",
    "Education": "Education Level",
    "Income": "Income Level",
}

GROUP_DESCRIPTIONS = {
    "Group 1: Health Conditions": "Core clinical and health-status indicators.",
    "Group 2: Lifestyle": "Lifestyle, diet, and healthcare-access variables.",
    "Group 3: Demographics": "Demographic information used by the model.",
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7fafc;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            background: #f1f5f9;
            border-right: 1px solid #dbe4ee;
        }

        [data-testid="stSidebar"] * {
            color: #102a43 !important;
        }

        h1, h2, h3, h4, h5, h6,
        p,
        label,
        .stCaption,
        [data-testid="stWidgetLabel"] {
            color: #102a43 !important;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dbe4ee;
            border-radius: 14px;
            padding: 0.75rem 0.9rem;
        }

        div[data-testid="stFormSubmitButton"] > button,
        .stButton > button {
            height: 2.85rem;
            border-radius: 12px;
            border: none;
            background: #0f4c81;
            color: #ffffff;
            font-weight: 700;
        }

        div[data-testid="stFormSubmitButton"] > button:hover,
        .stButton > button:hover {
            background: #0b3d68;
            color: #ffffff;
        }

        .simple-card {
            background: #ffffff;
            border: 1px solid #dbe4ee;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }

        .simple-title {
            color: #102a43;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .simple-text {
            color: #52606d;
            line-height: 1.6;
        }

        .result-box {
            border-radius: 16px;
            padding: 1rem 1.1rem;
            border: 1px solid #dbe4ee;
            margin-bottom: 1rem;
        }

        .result-box.low {
            background: #f0fdf4;
            border-left: 6px solid #15803d;
        }

        .result-box.medium {
            background: #fffbeb;
            border-left: 6px solid #c2410c;
        }

        .result-box.high {
            background: #fef2f2;
            border-left: 6px solid #b91c1c;
        }

        .result-badge {
            display: inline-block;
            margin-top: 0.55rem;
            padding: 0.3rem 0.68rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
        }

        .result-badge.low {
            background: #dcfce7;
            color: #166534;
        }

        .result-badge.medium {
            background: #fed7aa;
            color: #9a3412;
        }

        .result-badge.high {
            background: #fecaca;
            color: #991b1b;
        }

        .factor-card {
            border: 1px solid #dbe4ee;
            border-radius: 12px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.7rem;
        }

        .factor-card.risk {
            background: #fff1f2;
            border-left: 5px solid #be123c;
        }

        .factor-card.protective {
            background: #eff6ff;
            border-left: 5px solid #0369a1;
        }

        .factor-name {
            color: #102a43;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .factor-meta {
            color: #52606d;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_json_artifact(path_value: str) -> dict[str, Any] | list[Any] | None:
    path = Path(path_value)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def check_backend(backend_url: str) -> tuple[bool, str]:
    try:
        response = requests.get(f"{backend_url}/health", timeout=5)
        response.raise_for_status()
        payload = response.json()
        if payload.get("model_loaded"):
            return True, "Backend connected and model loaded."
        return False, "Backend connected, but the model is missing. Run `python main.py` first."
    except requests.RequestException as exc:
        return False, f"Backend connection failed: {exc}"


def fetch_metadata(backend_url: str) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{backend_url}/metadata", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def fetch_explanation_summary(backend_url: str) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{backend_url}/explain/summary", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def request_prediction_explanation(backend_url: str, payload: dict[str, int | float]) -> dict[str, Any]:
    response = requests.post(f"{backend_url}/explain", json=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def get_display_label(feature: str) -> str:
    return DISPLAY_LABELS.get(feature, feature)


def format_feature_value(feature: str, value: Any) -> str:
    if feature in OPTION_LABELS:
        numeric_value = int(float(value))
        return OPTION_LABELS[feature].get(numeric_value, str(numeric_value))

    numeric_value = float(value)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return f"{numeric_value:.1f}"


def get_risk_level(probability: float) -> tuple[str, str, str]:
    if probability < 0.35:
        return "Low Risk", "low", "This profile falls in the lower-risk range."
    if probability < 0.7:
        return "Medium Risk", "medium", "This profile shows a balanced mix of risk and protective signals."
    return "High Risk", "high", "This profile contains several strong diabetes risk indicators."


def validate_payload(payload: dict[str, int | float]) -> list[str]:
    errors: list[str] = []
    for feature, value in payload.items():
        lower, upper = VALIDATION_BOUNDS[feature]
        if not lower <= float(value) <= upper:
            errors.append(f"{get_display_label(feature)} must stay between {lower} and {upper}.")
    return errors


def render_select_field(feature: str) -> int:
    options = list(OPTION_LABELS[feature].keys())
    return st.selectbox(
        get_display_label(feature),
        options=options,
        index=options.index(DEFAULT_INPUTS[feature]),
        format_func=lambda value: OPTION_LABELS[feature][value],
        help=FIELD_HELP[feature],
    )


def render_numeric_field(feature: str) -> int | float:
    lower, upper = VALIDATION_BOUNDS[feature]
    default = DEFAULT_INPUTS[feature]

    if isinstance(default, float):
        return st.number_input(
            get_display_label(feature),
            min_value=float(lower),
            max_value=float(upper),
            value=float(default),
            step=0.1,
            format="%.1f",
            help=FIELD_HELP[feature],
        )

    return st.number_input(
        get_display_label(feature),
        min_value=int(lower),
        max_value=int(upper),
        value=int(default),
        step=1,
        help=FIELD_HELP[feature],
    )


def render_field(feature: str) -> int | float:
    if feature in OPTION_LABELS:
        return render_select_field(feature)
    return render_numeric_field(feature)


def build_payload() -> dict[str, int | float]:
    payload: dict[str, int | float] = {}
    for title, fields in FIELD_GROUPS:
        with st.expander(title, expanded=True):
            st.caption(GROUP_DESCRIPTIONS[title])
            columns = st.columns(2, gap="large")
            for index, feature in enumerate(fields):
                with columns[index % 2]:
                    payload[feature] = render_field(feature)
    return payload


def render_plot(title: str, path: Path, *, caption: str) -> None:
    with st.container(border=True):
        st.markdown(f"#### {title}")
        if path.exists():
            st.image(str(path), use_container_width=True)
            st.caption(caption)
        else:
            st.info(f"{title} will appear after the training script generates artifacts.")


def render_factor_cards(title: str, items: list[dict[str, Any]], *, tone: str, limit: int = 5) -> None:
    st.markdown(f"#### {title}")
    if not items:
        st.info("No explanation data is available yet.")
        return

    for item in items[:limit]:
        st.markdown(
            f"""
            <div class="factor-card {tone}">
                <div class="factor-name">{get_display_label(item["feature"])}</div>
                <div class="factor-meta">
                    Value: {format_feature_value(item["feature"], item["value"])}<br>
                    SHAP impact: {float(item["shap_value"]):+.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_feature_importance_table(items: list[dict[str, Any]], limit: int = 10) -> None:
    if not items:
        st.info("Global feature importance is not available yet.")
        return

    st.dataframe(
        [
            {
                "Feature": get_display_label(item["feature"]),
                "Mean |SHAP|": item.get("mean_abs_shap"),
                "Mean SHAP": item.get("mean_shap"),
            }
            for item in items[:limit]
        ],
        use_container_width=True,
        hide_index=True,
    )


def get_selected_model_metrics(training_report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not training_report:
        return None

    selected_model = training_report.get("selected_model")
    for model_entry in training_report.get("models", []):
        if model_entry.get("model_name") == selected_model:
            return model_entry
    return None


def render_model_metrics(training_report: dict[str, Any] | None) -> None:
    selected_metrics = get_selected_model_metrics(training_report)
    if not selected_metrics:
        st.info("Training metrics will appear here after `python main.py` generates the report.")
        return

    metric_columns = st.columns(5, gap="medium")
    metric_columns[0].metric("Accuracy", f"{selected_metrics['accuracy']:.4f}")
    metric_columns[1].metric("Precision", f"{selected_metrics['precision']:.4f}")
    metric_columns[2].metric("Recall", f"{selected_metrics['recall']:.4f}")
    metric_columns[3].metric("F1-score", f"{selected_metrics['f1_score']:.4f}")
    metric_columns[4].metric("ROC AUC", f"{selected_metrics['roc_auc']:.4f}")


def render_overview(metadata: dict[str, Any] | None, is_ready: bool) -> None:
    cols = st.columns(3, gap="medium")
    cols[0].metric("Dataset", metadata["dataset_used"] if metadata else "diabetes_brfss_binary.csv")
    cols[1].metric("Model", metadata["selected_model"] if metadata else "Random Forest")
    cols[2].metric("Backend", "Connected" if is_ready else "Offline")


def render_result_box(explanation: dict[str, Any]) -> None:
    probability = float(explanation["probability_diabetic"])
    probability_pct = round(probability * 100, 2)
    risk_label, risk_class, risk_text = get_risk_level(probability)

    st.markdown(
        f"""
        <div class="result-box {risk_class}">
            <div class="simple-title">Prediction Summary</div>
            <div class="simple-text">{risk_text}</div>
            <span class="result-badge {risk_class}">{risk_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(3, gap="medium")
    metric_cols[0].metric("Prediction", explanation["label"])
    metric_cols[1].metric("Probability", f"{probability_pct}%")
    metric_cols[2].metric("Model", explanation["selected_model"])
    st.progress(min(max(int(probability_pct), 0), 100))


def render_prediction_result(explanation: dict[str, Any]) -> None:
    render_result_box(explanation)

    result_cols = st.columns([1.2, 0.8], gap="large")
    with result_cols[0]:
        with st.container(border=True):
            st.markdown("#### Why This Prediction Was Made")
            st.write(explanation["explanation_text"])
            st.caption("Open the Explainability page for full SHAP-based factor details.")

    with result_cols[1]:
        interpretation = explanation["clinical_interpretation"]
        with st.container(border=True):
            st.markdown("#### Clinical Interpretation")
            st.write(interpretation["summary"])
            if interpretation["triggered_rules"]:
                st.markdown("**Triggered Rules**")
                for rule in interpretation["triggered_rules"]:
                    st.write(f"- {rule['message']}")
            else:
                st.info("No clinical rules were triggered for this profile.")


def render_prediction_page(
    backend_url: str,
    is_ready: bool,
    training_report: dict[str, Any] | None,
) -> None:
    st.subheader("Prediction")
    st.caption("Enter a patient profile and generate a diabetes screening prediction.")

    form_col, info_col = st.columns([1.7, 1], gap="large")

    with form_col:
        with st.form("prediction_form", clear_on_submit=False):
            payload = build_payload()
            validation_errors = validate_payload(payload)

            if validation_errors:
                for error in validation_errors:
                    st.error(error)

            submitted = st.form_submit_button(
                "Predict Diabetes Risk",
                use_container_width=True,
                disabled=(not is_ready) or bool(validation_errors),
            )

        if submitted:
            with st.spinner("Generating prediction..."):
                try:
                    explanation = request_prediction_explanation(backend_url, payload)
                except requests.HTTPError as exc:
                    detail = "Backend request failed."
                    try:
                        detail = exc.response.json().get("detail", detail)
                    except ValueError:
                        pass
                    st.error(detail)
                except requests.RequestException as exc:
                    st.error(f"Could not reach the backend API: {exc}")
                else:
                    st.session_state["latest_explanation"] = explanation
                    st.success("Prediction completed successfully.")

    with info_col:
        selected_metrics = get_selected_model_metrics(training_report)
        st.markdown(
            """
            <div class="simple-card">
                <div class="simple-title">How To Use</div>
                <div class="simple-text">
                    Fill the health, lifestyle, and demographic inputs, then click the prediction button.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="simple-card">
                <div class="simple-title">Note</div>
                <div class="simple-text">
                    This dashboard is an academic screening tool and not a medical diagnosis system.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if selected_metrics:
            with st.container(border=True):
                st.markdown("#### Model Snapshot")
                st.write("Current production model: `Random Forest`")
                st.write(f"Recall: `{selected_metrics['recall']:.4f}`")
                st.write(f"F1-score: `{selected_metrics['f1_score']:.4f}`")

    st.markdown("### Result")
    latest_explanation = st.session_state.get("latest_explanation")
    if latest_explanation:
        render_prediction_result(latest_explanation)
    else:
        st.info("Submit a patient profile to see the prediction result.")


def render_model_dashboard(summary: dict[str, Any] | None, training_report: dict[str, Any] | None) -> None:
    st.subheader("Model Dashboard")
    st.caption("Model performance and evaluation visuals for the deployed system.")

    render_model_metrics(training_report)

    with st.container(border=True):
        selected_model = summary["selected_model"] if summary else "Random Forest"
        selection_metric = summary["selection_metric"] if summary else "0.6 * cv_recall + 0.4 * cv_f1"
        st.markdown("#### Evaluation Summary")
        st.write(
            f"The deployed model is **{selected_model}**, selected using **{selection_metric}** to prioritize recall and F1-score."
        )

    plot_row_one = st.columns(2, gap="large")
    with plot_row_one[0]:
        render_plot(
            "Confusion Matrix",
            CONFUSION_MATRIX_PLOT_PATH,
            caption="Holdout-set confusion matrix for the final model.",
        )
    with plot_row_one[1]:
        render_plot(
            "ROC Curve",
            ROC_CURVE_PLOT_PATH,
            caption="Receiver operating characteristic curve for the final model.",
        )

    plot_row_two = st.columns(2, gap="large")
    with plot_row_two[0]:
        render_plot(
            "Feature Importance",
            FEATURE_IMPORTANCE_PLOT_PATH,
            caption="Global importance ranking derived from SHAP values.",
        )
    with plot_row_two[1]:
        render_plot(
            "Class Distribution",
            CLASS_DISTRIBUTION_PLOT_PATH,
            caption="Balanced BRFSS class distribution used for training.",
        )


def get_global_importance(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if summary and isinstance(summary.get("global_feature_importance"), list):
        return summary["global_feature_importance"]

    global_artifact = load_json_artifact(str(GLOBAL_EXPLANATION_PATH))
    if isinstance(global_artifact, list):
        return global_artifact
    return []


def get_local_explanation(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    latest_explanation = st.session_state.get("latest_explanation")
    if latest_explanation:
        return latest_explanation

    local_artifact = load_json_artifact(str(LOCAL_EXPLANATION_EXAMPLE_PATH))
    if isinstance(local_artifact, dict):
        return local_artifact

    if summary and isinstance(summary.get("example_local_explanation"), dict):
        return summary["example_local_explanation"]
    return None


def render_explainability_dashboard(summary: dict[str, Any] | None) -> None:
    st.subheader("Explainability")
    st.caption("Global SHAP behavior and detailed local explanation for a single prediction.")

    plot_cols = st.columns(2, gap="large")
    with plot_cols[0]:
        render_plot(
            "SHAP Summary Plot",
            SHAP_SUMMARY_PLOT_PATH,
            caption="Features pushing predictions higher or lower across the dataset.",
        )
    with plot_cols[1]:
        render_plot(
            "SHAP Global Importance",
            SHAP_BAR_PLOT_PATH,
            caption="Average absolute SHAP impact across the feature set.",
        )

    with st.container(border=True):
        st.markdown("#### Global Feature Importance")
        render_feature_importance_table(get_global_importance(summary), limit=10)

    local_explanation = get_local_explanation(summary)
    st.markdown("### Local Explanation")
    if not local_explanation:
        st.info("Run a prediction first to see a patient-level explanation.")
        return

    with st.container(border=True):
        st.markdown("#### Why This Prediction Was Made")
        st.write(
            local_explanation.get(
                "explanation_text",
                "The local explanation text is not available yet.",
            )
        )

    factor_cols = st.columns(2, gap="large")
    with factor_cols[0]:
        render_factor_cards(
            "Top 5 Risk Factors",
            local_explanation.get("top_positive_features", []),
            tone="risk",
            limit=5,
        )
    with factor_cols[1]:
        render_factor_cards(
            "Top 5 Protective Factors",
            local_explanation.get("top_negative_features", []),
            tone="protective",
            limit=5,
        )


def render_sidebar(
    default_backend_url: str,
) -> tuple[str, str, bool, dict[str, Any] | None, dict[str, Any] | None]:
    with st.sidebar:
        st.title("Dashboard")
        page = st.radio("Page", PAGE_OPTIONS, label_visibility="collapsed")
        st.divider()

        st.markdown("**Backend**")
        backend_url = st.text_input("Backend URL", value=default_backend_url, key="backend_url")
        is_ready, status_message = check_backend(backend_url)
        metadata = fetch_metadata(backend_url) if is_ready else None
        summary = fetch_explanation_summary(backend_url) if is_ready else None

        if is_ready:
            st.success(status_message)
        else:
            st.warning(status_message)

        st.caption("Prediction: enter profile and generate result.")
        st.caption("Dashboard: review performance metrics and plots.")
        st.caption("Explainability: inspect global and local SHAP outputs.")

    return page, backend_url, is_ready, metadata, summary


def main() -> None:
    st.set_page_config(
        page_title="Diabetes Prediction System",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    training_report = load_json_artifact(str(TRAINING_REPORT_PATH))
    if not isinstance(training_report, dict):
        training_report = None

    page, backend_url, is_ready, metadata, summary = render_sidebar(DEFAULT_BACKEND_URL)

    st.title("Diabetes Prediction System")
    st.caption("Clean dashboard for BRFSS-based diabetes risk prediction, evaluation, and explainability.")
    render_overview(metadata, is_ready)
    st.divider()

    if page == "Prediction":
        render_prediction_page(backend_url, is_ready, training_report)
    elif page == "Model Dashboard":
        render_model_dashboard(summary, training_report)
    else:
        render_explainability_dashboard(summary)


if __name__ == "__main__":
    main()
