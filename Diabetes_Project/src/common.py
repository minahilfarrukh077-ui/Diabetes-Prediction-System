from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
EXPLANATIONS_DIR = ARTIFACTS_DIR / "explanations"
DATASET_DIR = BASE_DIR / "dataset"
LEGACY_DATASET_DIR = DATASET_DIR / "legacy"
ARCHIVE_DIR = BASE_DIR / "archive (5)"

MODEL_PATH = ARTIFACTS_DIR / "diabetes_model.joblib"
TRAINING_REPORT_PATH = ARTIFACTS_DIR / "training_report.json"
DATA_AUDIT_PATH = ARTIFACTS_DIR / "data_audit.json"
MODEL_COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.csv"
GLOBAL_EXPLANATION_PATH = EXPLANATIONS_DIR / "global_feature_importance.json"
LOCAL_EXPLANATION_EXAMPLE_PATH = EXPLANATIONS_DIR / "example_local_explanation.json"
CLASS_DISTRIBUTION_PLOT_PATH = PLOTS_DIR / "class_distribution.png"
CONFUSION_MATRIX_PLOT_PATH = PLOTS_DIR / "confusion_matrix.png"
ROC_CURVE_PLOT_PATH = PLOTS_DIR / "roc_curve.png"
FEATURE_IMPORTANCE_PLOT_PATH = PLOTS_DIR / "feature_importance.png"
SHAP_SUMMARY_PLOT_PATH = PLOTS_DIR / "shap_summary.png"
SHAP_BAR_PLOT_PATH = PLOTS_DIR / "shap_global_bar.png"

PRIMARY_DATASET = DATASET_DIR / "diabetes_brfss_binary.csv"

TARGET_COLUMN = "Diabetes_binary"
FEATURE_COLUMNS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]
NUMERIC_COLUMNS = FEATURE_COLUMNS.copy()

VALIDATION_BOUNDS = {
    "HighBP": (0, 1),
    "HighChol": (0, 1),
    "CholCheck": (0, 1),
    "BMI": (12.0, 98.0),
    "Smoker": (0, 1),
    "Stroke": (0, 1),
    "HeartDiseaseorAttack": (0, 1),
    "PhysActivity": (0, 1),
    "Fruits": (0, 1),
    "Veggies": (0, 1),
    "HvyAlcoholConsump": (0, 1),
    "AnyHealthcare": (0, 1),
    "NoDocbcCost": (0, 1),
    "GenHlth": (1, 5),
    "MentHlth": (0, 30),
    "PhysHlth": (0, 30),
    "DiffWalk": (0, 1),
    "Sex": (0, 1),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),
}

DEFAULT_INPUTS = {
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
}

FIELD_LABELS = {
    "HighBP": "HighBP (High Blood Pressure)",
    "HighChol": "HighChol (High Cholesterol)",
    "CholCheck": "CholCheck (Cholesterol Checked in 5 Years)",
    "BMI": "BMI",
    "Smoker": "Smoker",
    "Stroke": "Stroke History",
    "HeartDiseaseorAttack": "HeartDiseaseorAttack",
    "PhysActivity": "PhysActivity",
    "Fruits": "Fruits",
    "Veggies": "Veggies",
    "HvyAlcoholConsump": "HvyAlcoholConsump",
    "AnyHealthcare": "AnyHealthcare",
    "NoDocbcCost": "NoDocbcCost",
    "GenHlth": "GenHlth (General Health)",
    "MentHlth": "MentHlth (Mental Health Days)",
    "PhysHlth": "PhysHlth (Physical Health Days)",
    "DiffWalk": "DiffWalk",
    "Sex": "Sex",
    "Age": "Age Category",
    "Education": "Education Level",
    "Income": "Income Level",
}

FIELD_HELP = {
    "HighBP": "BRFSS-coded binary field: 1 means the person has high blood pressure.",
    "HighChol": "BRFSS-coded binary field: 1 means the person has high cholesterol.",
    "CholCheck": "1 means cholesterol was checked within the last 5 years.",
    "BMI": "Body Mass Index from the BRFSS dataset.",
    "Smoker": "1 means the person has smoked at least 100 cigarettes in life.",
    "Stroke": "1 means the person has a stroke history.",
    "HeartDiseaseorAttack": "1 means coronary heart disease or myocardial infarction history.",
    "PhysActivity": "1 means physical activity was reported during the past 30 days.",
    "Fruits": "1 means fruit consumption was reported.",
    "Veggies": "1 means vegetable consumption was reported.",
    "HvyAlcoholConsump": "1 means heavy alcohol consumption was reported.",
    "AnyHealthcare": "1 means the person has healthcare coverage.",
    "NoDocbcCost": "1 means medical cost stopped the person from seeing a doctor.",
    "GenHlth": "General health rating from 1 to 5.",
    "MentHlth": "Number of poor mental health days in the last 30 days.",
    "PhysHlth": "Number of poor physical health days in the last 30 days.",
    "DiffWalk": "1 means difficulty walking or climbing stairs.",
    "Sex": "BRFSS binary sex code.",
    "Age": "Age bucket encoded from 1 to 13.",
    "Education": "Education bucket encoded from 1 to 6.",
    "Income": "Income bucket encoded from 1 to 8.",
}

BOOLEAN_CHOICES = {0: "No", 1: "Yes"}
SEX_CHOICES = {0: "Female", 1: "Male"}
GENERAL_HEALTH_CHOICES = {
    1: "Excellent",
    2: "Very good",
    3: "Good",
    4: "Fair",
    5: "Poor",
}
AGE_CHOICES = {
    1: "18-24",
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65-69",
    11: "70-74",
    12: "75-79",
    13: "80+",
}
EDUCATION_CHOICES = {
    1: "Never attended school or kindergarten only",
    2: "Grades 1-8",
    3: "Grades 9-11",
    4: "Grade 12 or GED",
    5: "College 1-3 years",
    6: "College 4+ years",
}
INCOME_CHOICES = {
    1: "Less than $10,000",
    2: "$10,000-$15,000",
    3: "$15,000-$20,000",
    4: "$20,000-$25,000",
    5: "$25,000-$35,000",
    6: "$35,000-$50,000",
    7: "$50,000-$75,000",
    8: "$75,000 or more",
}

OPTION_LABELS = {
    "HighBP": BOOLEAN_CHOICES,
    "HighChol": BOOLEAN_CHOICES,
    "CholCheck": BOOLEAN_CHOICES,
    "Smoker": BOOLEAN_CHOICES,
    "Stroke": BOOLEAN_CHOICES,
    "HeartDiseaseorAttack": BOOLEAN_CHOICES,
    "PhysActivity": BOOLEAN_CHOICES,
    "Fruits": BOOLEAN_CHOICES,
    "Veggies": BOOLEAN_CHOICES,
    "HvyAlcoholConsump": BOOLEAN_CHOICES,
    "AnyHealthcare": BOOLEAN_CHOICES,
    "NoDocbcCost": BOOLEAN_CHOICES,
    "DiffWalk": BOOLEAN_CHOICES,
    "Sex": SEX_CHOICES,
    "GenHlth": GENERAL_HEALTH_CHOICES,
    "Age": AGE_CHOICES,
    "Education": EDUCATION_CHOICES,
    "Income": INCOME_CHOICES,
}

FIELD_GROUPS = [
    (
        "Group 1: Health Conditions",
        [
            "HighBP",
            "HighChol",
            "CholCheck",
            "BMI",
            "Stroke",
            "HeartDiseaseorAttack",
            "GenHlth",
            "MentHlth",
            "PhysHlth",
            "DiffWalk",
        ],
    ),
    (
        "Group 2: Lifestyle",
        [
            "Smoker",
            "PhysActivity",
            "Fruits",
            "Veggies",
            "HvyAlcoholConsump",
            "AnyHealthcare",
            "NoDocbcCost",
        ],
    ),
    (
        "Group 3: Demographics",
        ["Sex", "Age", "Education", "Income"],
    ),
]


def resolve_dataset_path() -> Path:
    if PRIMARY_DATASET.exists():
        return PRIMARY_DATASET
    raise FileNotFoundError(
        f"Primary dataset was not found at {PRIMARY_DATASET}. "
        "Copy the BRFSS dataset into the dataset folder before training."
    )
