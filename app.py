import io
import json
from pathlib import Path
import sys
import importlib.util
import streamlit as st

st.write("Python version:", sys.version)
st.write("Python location:", sys.executable)
st.write("Joblib available:", importlib.util.find_spec("joblib"))

import joblib
import pandas as pd


BASE_DIR = Path(__file__).parent
classifier_bytes = b"".join(
    (BASE_DIR / f"classifier.part{i}").read_bytes()
    for i in range(1, 4)
)

CLASSIFIER = joblib.load(io.BytesIO(classifier_bytes))
REGRESSOR = joblib.load(BASE_DIR / "regressor.pkl")
with open(BASE_DIR / "feature_metadata.json", "r", encoding="utf-8") as f:
    META = json.load(f)

st.set_page_config(
    page_title="Healthcare Disease Prediction",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Healthcare Disease Prediction")
st.write(
    "Enter patient health information to explore the predictions from the "
    "project's tuned Random Forest models."
)
st.info(
    "Educational project only. Predictions are not medical diagnoses and "
    "must not be used to make treatment decisions."
)

def pretty(name):
    return name.replace("_", " ").replace("HbA1c", "HbA1c").title()

def collect_inputs(task_key, feature_key):
    fields = META[task_key]
    values = {}
    with st.form(f"{task_key}_form"):
        st.subheader("Patient information")
        cols = st.columns(2)
        for i, field in enumerate(fields):
            target = cols[i % 2]
            name = field["name"]
            label = pretty(name)
            with target:
                if field["type"] == "number":
                    lo = float(field["min"])
                    hi = float(field["max"])
                    default = float(field["default"])
                    if lo == hi:
                        lo -= 1
                        hi += 1
                    default = min(max(default, lo), hi)
                    step = 1.0 if float(default).is_integer() else 0.1
                    values[name] = st.number_input(
                        label,
                        min_value=lo,
                        max_value=hi,
                        value=default,
                        step=step,
                        key=f"{task_key}_{name}",
                    )
                else:
                    options = field["options"]
                    default = field["default"] if field["default"] in options else options[0]
                    index = options.index(default)
                    values[name] = st.selectbox(
                        label,
                        options=options,
                        index=index,
                        key=f"{task_key}_{name}",
                    )
        submitted = st.form_submit_button("Generate prediction", type="primary")
    return values, submitted

def prepare_row(values, feature_names):
    row = dict(values)
    # Recreate the engineered features used during model training.
    if "pulse_pressure" in feature_names:
        row["pulse_pressure"] = (
            float(row["systolic_bp"]) - float(row["diastolic_bp"])
        )
    if "cholesterol_ratio" in feature_names:
        hdl = float(row["hdl"])
        row["cholesterol_ratio"] = (
            float(row["cholesterol"]) / hdl if hdl != 0 else None
        )
    return pd.DataFrame(
        [[row.get(col) for col in feature_names]],
        columns=feature_names,
    )

tab1, tab2 = st.tabs(["Health Classification", "BMI Prediction"])

with tab1:
    st.subheader("Predict Normal or Abnormal")
    st.caption(
        "The Random Forest Classifier estimates the patient's health category "
        "from the health indicators entered below."
    )
    values, submitted = collect_inputs("classification", "classification_features")
    if submitted:
        input_df = prepare_row(values, META["classification_features"])
        prediction = CLASSIFIER.predict(input_df)[0]
        probabilities = CLASSIFIER.predict_proba(input_df)[0]
        classes = list(CLASSIFIER.named_steps["model"].classes_)
        abnormal_probability = float(probabilities[classes.index("Abnormal")])
        if prediction == "Abnormal":
            st.error(f"Model prediction: {prediction}")
        else:
            st.success(f"Model prediction: {prediction}")
        st.metric("Estimated probability of Abnormal", f"{abnormal_probability:.1%}")
        st.caption("This probability is a model output, not a clinical risk assessment.")

with tab2:
    st.subheader("Predict BMI")
    st.caption(
        "The Random Forest Regressor estimates a numerical BMI value using "
        "the other patient health indicators."
    )
    values, submitted = collect_inputs("regression", "regression_features")
    if submitted:
        input_df = prepare_row(values, META["regression_features"])
        bmi_prediction = float(REGRESSOR.predict(input_df)[0])
        st.metric("Predicted BMI", f"{bmi_prediction:.2f}")
        st.caption(
            "BMI is a model estimate. It should be interpreted with other "
            "health information by a qualified healthcare professional."
        )

st.divider()
st.caption(
    "Models: tuned Random Forest Classifier and Random Forest Regressor. "
    "The application uses the feature columns and preprocessing pipeline "
    "from the project notebook."
)
