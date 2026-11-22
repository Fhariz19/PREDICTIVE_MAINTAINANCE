import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

from dashboard_page import render_dashboard

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)

# Load trained model
MODEL_PATH = Path(__file__).parent / "models" / "final_predictive_maintenance_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model_data = load_model()
model = model_data["model"]
threshold = model_data["threshold"]
features = model_data["features"]

# Application title
st.title("⚙️ Predictive Maintenance System")
st.write("Machine Failure Prediction using Machine Learning")

st.divider()

# Sidebar

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071923 0%, #0B2633 100%);
}

[data-testid="stSidebar"] * {
    color: #E5F3F7;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background-color: #102F3D;
    border: 1px solid #254B5A;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    transition: 0.2s ease;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background-color: #164354;
    border-color: #22D3EE;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background-color: #164354;
    border: 1px solid #22D3EE;
}

[data-testid="stSidebar"] [data-testid="stRadio"] input {
    accent-color: #22D3EE;
}
</style>
""", unsafe_allow_html=True)

# Sidebar Branding
st.sidebar.markdown("""
<div style="padding: 15px 5px 20px 5px;">
    <h2 style="color:#22D3EE; margin-bottom:0;">⚙ PREDICTIX</h2>
    <h3 style="color:#E5F3F7; margin-top:0;">PREDICTIVE MAINTENANCE</h3>
    <div style="height:3px; width:55px; background:#22D3EE; border-radius:5px;"></div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### MAIN MENU")

page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Machine Failure Prediction", "About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("""
<div style="position:fixed; bottom:20px; font-size:11px; color:#8BAAB5;">
    AI4I 2020 · PREDICTIVE ANALYTICS
</div>
""", unsafe_allow_html=True)

if page == "Dashboard":
    render_dashboard(threshold)

elif page == "Machine Failure Prediction":

    # Observed ranges in the AI4I 2020 training dataset
    TRAINING_RANGES = {
        "Air temperature [K]": (295.3, 304.5),
        "Process temperature [K]": (305.7, 313.8),
        "Rotational speed [rpm]": (1168, 2886),
        "Torque [Nm]": (3.8, 76.6),
        "Tool wear [min]": (0, 253),
    }

    with st.form("prediction_form"):

        st.subheader("Machine Operating Conditions")

        col1, col2 = st.columns(2)

        with col1:
            air_temp = st.number_input(
                "Air Temperature [K]",
                min_value=0.0,
                value=300.0,
                step=0.1
            )

            process_temp = st.number_input(
                "Process Temperature [K]",
                min_value=0.0,
                value=310.0,
                step=0.1
            )

            speed = st.number_input(
                "Rotational Speed [rpm]",
                min_value=0,
                value=1500,
                step=10
            )

        with col2:
            torque = st.number_input(
                "Torque [Nm]",
                min_value=0.0,
                value=40.0,
                step=0.1
            )

            wear = st.number_input(
                "Tool Wear [min]",
                min_value=0,
                value=100,
                step=1
            )

            machine_type = st.selectbox(
                "Machine Type",
                ["H", "L", "M"]
            )

        submitted = st.form_submit_button(
            "Predict Machine Failure",
            use_container_width=True
        )

    if submitted:

        # BASIC INPUT VALIDATION

        errors = []

        if air_temp <= 0 or process_temp <= 0:
            errors.append("Temperature must be greater than zero.")

        if speed <= 0:
            errors.append("Rotational speed must be greater than zero.")

        if torque <= 0:
            errors.append("Torque must be greater than zero.")

        if wear < 0:
            errors.append("Tool wear cannot be negative.")

        if process_temp < air_temp:
            errors.append(
                "Process temperature should not be lower than air temperature."
            )

        if errors:
            for error in errors:
                st.error(error)

        else:

            # TRAINING DATA RANGE CHECK

            entered_values = {
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": wear,
            }

            out_of_range = []

            for name, value in entered_values.items():

                minimum, maximum = TRAINING_RANGES[name]

                if value < minimum or value > maximum:
                    out_of_range.append(
                        f"{name}: entered {value}, "
                        f"training range {minimum}–{maximum}"
                    )

            if out_of_range:
                st.warning(
                    "Some inputs are outside the observed training-data "
                    "range. Predictions may be less reliable for these values."
                )

                for item in out_of_range:
                    st.caption(f"• {item}")

            # FEATURE ENGINEERING

            temperature_difference = process_temp - air_temp
            machine_power = torque * speed
            wear_torque_interaction = wear * torque

            input_data = pd.DataFrame([{
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": wear,
                "Temperature Difference": temperature_difference,
                "Machine Power": machine_power,
                "Wear Torque Interaction": wear_torque_interaction,
                "Type_L": int(machine_type == "L"),
                "Type_M": int(machine_type == "M")
            }], columns=features)

            # MODEL PREDICTION

            probability = model.predict_proba(input_data)[0][1]
            prediction = probability >= threshold

            st.divider()
            st.subheader("Prediction Result")

            result_col1, result_col2 = st.columns(2)

            result_col1.metric(
                "Failure Probability",
                f"{probability:.2%}"
            )

            result_col2.metric(
                "Decision Threshold",
                f"{threshold:.2%}"
            )

            st.progress(float(probability))

            # RISK INTERPRETATION

            if probability < threshold:
                risk_level = "Low"
                st.success(
                    "No Machine Failure Predicted. "
                    "Continue routine monitoring."
                )

            elif probability < 0.70:
                risk_level = "Moderate"
                st.warning(
                    "Potential Machine Failure Detected. "
                    "Further machine inspection is recommended."
                )

            else:
                risk_level = "High"
                st.error(
                    "High Failure Risk Detected. "
                    "Prioritise machine inspection and maintenance."
                )

            st.markdown(f"**Risk Classification: {risk_level}**")

            st.caption(
                "Risk labels are explanatory categories: Low is below "
                "the model threshold, Moderate is from the threshold "
                "to below 70%, and High is 70% or above. "
                "The model's actual failure decision still uses its "
                "original threshold."
            )

            # MAINTENANCE RECOMMENDATIONS

            st.divider()
            st.subheader("Maintenance Recommendations")

            if prediction:

                st.write(
                    "The model indicates potential machine failure. "
                    "Consider the following inspection actions:"
                )

                recommendations = [
                    "Inspect machine condition before continued operation.",
                    "Review torque and rotational speed readings.",
                    "Check operating temperature and heat dissipation.",
                    "Review tool wear and consider tool replacement if necessary.",
                    "Record inspection findings for future maintenance planning."
                ]

                if wear >= 200:
                    recommendations.insert(
                        0,
                        "High tool wear reading: prioritise tool-condition inspection."
                    )

                if torque >= 50:
                    recommendations.append(
                        "Review high torque conditions and machine load."
                    )

                for recommendation in recommendations:
                    st.markdown(f"- {recommendation}")

            else:

                st.write(
                    "The model does not currently indicate machine failure."
                )

                st.markdown(
                    "- Continue routine machine monitoring.\n"
                    "- Maintain regular sensor checks.\n"
                    "- Follow the scheduled preventive maintenance plan."
                )

            st.caption(
                "Recommendations are general inspection suggestions, "
                "not confirmed machine diagnoses."
            )


elif page == "About":

    # Page Header
    st.header("About PredictiX")
    st.caption(
        "An intelligent machine failure prediction system "
        "powered by machine learning."
    )

    st.divider()

    # Project Overview
    st.subheader("Project Overview")

    st.write("""
    **PredictiX** is a Predictive Maintenance System developed to
    identify potential machine failures using machine learning.

    The system analyses machine operating conditions, including
    air temperature, process temperature, rotational speed, torque,
    and tool wear, to estimate the probability of machine failure.

    By providing early failure risk predictions and maintenance
    recommendations, PredictiX aims to support better machine
    monitoring and preventive maintenance planning.
    """)

    st.divider()

    # Project Objectives
    st.subheader("Project Objectives")

    obj1, obj2, obj3 = st.columns(3)

    with obj1:
        st.markdown("### 🎯 Early Detection")
        st.write(
            "Identify potential machine failures based on "
            "operational data."
        )

    with obj2:
        st.markdown("### 🤖 Machine Learning")
        st.write(
            "Develop a predictive model to estimate "
            "machine failure probability."
        )

    with obj3:
        st.markdown("### 🔧 Maintenance Support")
        st.write(
            "Provide risk classifications and recommendations "
            "to support maintenance decisions."
        )

    st.divider()

    # Dataset Information
    st.subheader("Dataset Information")

    data_col1, data_col2 = st.columns(2)

    with data_col1:
        st.metric("Total Records", "10,000")
        st.metric("Input Features", len(features))

    with data_col2:
        st.metric("Target Variable", "Machine Failure")
        st.metric("Dataset", "AI4I 2020")

    st.write(
        "The system uses the AI4I 2020 Predictive Maintenance Dataset "
        "containing machine operating conditions and failure information."
    )

    st.markdown(
        "[View Dataset on Kaggle](https://www.kaggle.com/datasets/"
        "stephanmatzka/predictive-maintenance-dataset-ai4i-2020)"
    )

    st.divider()

    # Machine Learning Model
    st.subheader("Machine Learning Model")

    st.markdown("""
    **Selected Model: Tuned Random Forest Classifier**

    Random Forest was selected as the final model following model
    comparison and hyperparameter tuning.

    The model uses a decision threshold of **39.59%** to classify
    whether a machine is predicted to experience failure.
    """)

    st.divider()

    # Model Performance
    st.subheader("Final Model Performance")

    st.caption("Evaluation results on the independent test dataset.")

    metric1, metric2, metric3 = st.columns(3)
    metric4, metric5, metric6 = st.columns(3)

    metric1.metric("Accuracy", "98.80%")
    metric2.metric("Precision", "80.00%")
    metric3.metric("Recall", "86.27%")

    metric4.metric("F1-Score", "83.02%")
    metric5.metric("ROC-AUC", "98.65%")
    metric6.metric("PR-AUC", "90.30%")

    st.info(
        "The final model achieved 86.27% recall, identifying "
        "44 out of 51 actual machine failures in the test dataset."
    )

    st.divider()

    # Team Information
    st.subheader("Project Team")

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #102F3D, #164354);
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #22D3EE;
    ">
        <h2 style="color:#22D3EE; margin-bottom:5px;">⚙ PredictiX</h2>
        <p style="color:#E5F3F7; margin:0;">
            Predictive Maintenance System
        </p>
        <p style="color:#9ABBC5; margin-top:12px; font-size:13px;">
            Developed as an academic machine learning project.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.caption(
        "PredictiX | AI4I 2020 Predictive Maintenance Project"
    )
