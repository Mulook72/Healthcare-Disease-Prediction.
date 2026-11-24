import io
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Configure the Streamlit page before displaying anything.
st.set_page_config(
    page_title="Healthcare Disease Prediction",
    page_icon="🩺",
    layout="wide",
)

BASE_DIR = Path(__file__).parent


@st.cache_resource
def load_models():
    """Load the classifier assembled from its three uploaded binary parts."""
    part_files = [
        BASE_DIR / "classifier.part1",
        BASE_DIR / "classifier.part2",
        BASE_DIR / "classifier.part3",
    ]

    missing_parts = [path.name for path in part_files if not path.is_file()]
    if missing_parts:
        raise FileNotFoundError(
            "Missing classifier file part(s): " + ", ".join(missing_parts)
        )

    classifier_bytes = b"".join(path.read_bytes() for path in part_files)
    classifier = joblib.load(io.BytesIO(classifier_bytes))

    regressor_path = BASE_DIR / "regressor.pkl"
    metadata_path = BASE_DIR / "feature_metadata.json"

    if not regressor_path.is_file():
        raise FileNotFoundError("regressor.pkl was not found beside app.py.")
    if not metadata_path.is_file():
        raise FileNotFoundError("feature_metadata.json was not found beside app.py.")

    regressor = joblib.load(regressor_path)

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    return classifier, regressor, metadata


try:
    CLASSIFIER, REGRESSOR, META = load_models()
except Exception as error:
    st.error("The prediction models could not be loaded.")
    st.exception(error)
    st.stop()


st.title("🩺 Healthcare Disease Prediction")
st.write(
    "Enter patient health information to explore predictions from the "
    "project's tuned Random Forest models."
)
st.info(
    "Educational project only. Predictions are not medical diagnoses and "
    "must not be used to make treatment decisions."
)


def pretty(name):
    return name.replace("_", " ").title()


def collect_inputs(task_key):
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
                    step = 1.0 if default.is_integer() else 0.1

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
                    default = (
                        field["default"]
                        if field["default"] in options
                        else options[0]
                    )
                    values[name] = st.selectbox(
                        label,
                        options=options,
                        index=options.index(default),
                        key=f"{task_key}_{name}",
                    )

        submitted = st.form_submit_button(
            "Generate prediction",
            type="primary",
        )

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
        [[row.get(column) for column in feature_names]],
        columns=feature_names,
    )


tab1, tab2 = st.tabs(["Health Classification", "BMI Prediction"])

with tab1:
    st.subheader("Predict Normal or Abnormal")
    st.caption(
        "The Random Forest Classifier estimates the patient's health category "
        "from the health indicators entered below."
    )

    values, submitted = collect_inputs("classification")

    if submitted:
        input_df = prepare_row(
            values,
            META["classification_features"],
        )

        prediction = CLASSIFIER.predict(input_df)[0]
        probabilities = CLASSIFIER.predict_proba(input_df)[0]
        classes = list(CLASSIFIER.named_steps["model"].classes_)

        if "Abnormal" in classes:
            abnormal_probability = float(
                probabilities[classes.index("Abnormal")]
            )
            st.metric(
                "Estimated probability of Abnormal",
                f"{abnormal_probability:.1%}",
            )
        else:
            st.warning(
                "The classifier output does not contain an 'Abnormal' class."
            )

        if prediction == "Abnormal":
            st.error(f"Model prediction: {prediction}")
        else:
            st.success(f"Model prediction: {prediction}")

        st.caption(
            "This probability is a model output, not a clinical risk assessment."
        )


with tab2:
    st.subheader("Predict BMI")
    st.caption(
        "The Random Forest Regressor estimates a numerical BMI value using "
        "the other patient health indicators."
    )

    values, submitted = collect_inputs("regression")

    if submitted:
        input_df = prepare_row(
            values,
            META["regression_features"],
        )

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
