# Diabetes Prediction Project

Python diabetes prediction system with:

- a Python machine learning training pipeline
- a Python backend API built with FastAPI
- a Python frontend UI built with Streamlit
- a real BRFSS diabetes dataset
- SHAP explainability and evaluation plots
- evaluation artifacts and VS Code configuration

## Current Dataset

The new primary dataset is:

- [dataset/diabetes_brfss_binary.csv](/Users/eshafarrukh/Downloads/Diabetes_Project/dataset/diabetes_brfss_binary.csv)

It was migrated from:

- [archive (5)/diabetes_binary_5050split_health_indicators_BRFSS2015.csv](/Users/eshafarrukh/Downloads/Diabetes_Project/archive%20(5)/diabetes_binary_5050split_health_indicators_BRFSS2015.csv)

Important note:

- the `archive (5)` folder does not contain a CBC lab dataset
- it contains BRFSS diabetes survey datasets
- the runtime now uses only `dataset/diabetes_brfss_binary.csv`
- `archive (5)` is kept only as a reference source

## Dataset Selection Decision

Three datasets were found in `archive (5)`:

1. `diabetes_012_health_indicators_BRFSS2015.csv`
2. `diabetes_binary_health_indicators_BRFSS2015.csv`
3. `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`

The selected primary dataset is the 50/50 binary BRFSS file because:

- it already matches the required binary classification task
- it avoids remapping the `0/1/2` multiclass target
- it has balanced classes, which improves F1-score and recall for diabetes screening
- it keeps the same health-indicator feature set used across the BRFSS files

## Project Structure

```text
Diabetes_Project/
  app.py
  api.py
  main.py
  dataset/
    diabetes_brfss_binary.csv
    legacy/
      diabetes_synthetic.csv
      diabetes_synthetic_duplicate.csv
  archive (5)/
  artifacts/
    diabetes_model.joblib
    data_audit.json
    model_comparison.csv
    training_report.json
    legacy/
  src/
    backend/
    frontend/
    ml/
    common.py
  tests/
```

## Features Used

The current feature list exactly matches the BRFSS dataset:

- `HighBP`
- `HighChol`
- `CholCheck`
- `BMI`
- `Smoker`
- `Stroke`
- `HeartDiseaseorAttack`
- `PhysActivity`
- `Fruits`
- `Veggies`
- `HvyAlcoholConsump`
- `AnyHealthcare`
- `NoDocbcCost`
- `GenHlth`
- `MentHlth`
- `PhysHlth`
- `DiffWalk`
- `Sex`
- `Age`
- `Education`
- `Income`

Target column:

- `Diabetes_binary`

## Machine Learning Algorithms

The training pipeline compares these three required algorithms:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## Training Logic

- the BRFSS dataset is loaded from `dataset/diabetes_brfss_binary.csv`
- missing values are handled with median imputation
- numerical features are standardized inside the sklearn pipeline
- target values are validated as `0 = non-diabetic` and `1 = diabetic`
- models are ranked by a healthcare-focused selection score:

```text
0.6 * cv_recall + 0.4 * cv_f1
```

- this gives extra weight to recall while still rewarding balanced performance

## Train The Model

```bash
python main.py
```

Training produces:

- `artifacts/diabetes_model.joblib`
- `artifacts/training_report.json`
- `artifacts/data_audit.json`
- `artifacts/model_comparison.csv`

Note:

- no separate scaler file is required
- the saved model artifact already contains preprocessing and the classifier in one pipeline

## Explainability and Visualization

Training now also generates:

- SHAP global feature importance JSON
- example local prediction explanation JSON
- confusion matrix plot
- ROC curve plot
- class distribution plot
- feature importance plot
- SHAP summary plot
- SHAP global bar plot

Saved locations:

- `artifacts/explanations/`
- `artifacts/plots/`

## Run The Backend

```bash
uvicorn api:app --reload
```

Available endpoints:

- `GET /health`
- `GET /metadata`
- `POST /predict`
- `POST /explain`
- `GET /explain/summary`

## Run The Frontend

```bash
streamlit run app.py
```

Optional environment variable:

```bash
export BACKEND_URL=http://127.0.0.1:8000
```

## Example Prediction Request

```json
{
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
  "Income": 6
}
```

## Legacy Data

The old synthetic datasets are no longer used in runtime:

- [dataset/legacy/diabetes_synthetic.csv](/Users/eshafarrukh/Downloads/Diabetes_Project/dataset/legacy/diabetes_synthetic.csv)
- [dataset/legacy/diabetes_synthetic_duplicate.csv](/Users/eshafarrukh/Downloads/Diabetes_Project/dataset/legacy/diabetes_synthetic_duplicate.csv)

## Deployment Notes

- Frontend: Streamlit Community Cloud
- Backend: Render or Railway
- Local demo: Streamlit + FastAPI on the same machine

## Important Limitation

This system is a diabetes risk screening project built from BRFSS survey indicators. It is not a clinical diagnostic tool and must not be used as medical advice.
