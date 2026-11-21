
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

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
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Machine Failure Prediction", "About"]
)

if page == "Dashboard":
    st.header("Dashboard Overview")
    st.info("Dashboard analytics will be developed by the team.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Records", "10,000")
    col2.metric("Selected Model", "Random Forest")
    col3.metric("Decision Threshold", f"{threshold:.4f}")

elif page == "Machine Failure Prediction":
    st.header("Machine Failure Prediction")
    st.write("Enter machine operating conditions below.")

    with st.form("prediction_form"):
        air_temp = st.number_input("Air Temperature [K]", value=300.0)
        process_temp = st.number_input("Process Temperature [K]", value=310.0)
        speed = st.number_input("Rotational Speed [rpm]", value=1500)
        torque = st.number_input("Torque [Nm]", value=40.0)
        wear = st.number_input("Tool Wear [min]", value=100)

        machine_type = st.selectbox("Machine Type", ["H", "L", "M"])

        submitted = st.form_submit_button("Predict Failure")

    
    if submitted:
        # Input validation
        if air_temp <= 0 or process_temp <= 0:
            st.error("Temperature must be greater than zero.")
        elif speed <= 0:
            st.error("Rotational speed must be greater than zero.")
        elif torque <= 0:
            st.error("Torque must be greater than zero.")
        elif wear < 0:
            st.error("Tool wear cannot be negative.")
        elif process_temp < air_temp:
            st.error("Process temperature should not be lower than air temperature.")
        else:
            # Feature engineering
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

            # Prediction
            probability = model.predict_proba(input_data)[0][1]
            prediction = probability >= threshold

            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            col1.metric(
                "Failure Probability",
                f"{probability:.4%}"
            )

            col2.metric(
                "Decision Threshold",
                f"{threshold:.2%}"
            )

            st.progress(float(probability))

            if prediction:
                st.error(
                    "⚠️ Potential Machine Failure Detected. "
                    "Maintenance inspection is recommended."
                )
            else:
                st.success(
                    "✅ No Machine Failure Predicted. "
                    "Continue routine monitoring."
                )


elif page == "About":
    st.header("About This Project")
    st.write("""
    This academic project uses the AI4I 2020 Predictive Maintenance Dataset
    and a tuned Random Forest model to predict machine failures.
    """)
    st.write(f"Model threshold: {threshold}")
    st.write(f"Number of input features: {len(features)}")
