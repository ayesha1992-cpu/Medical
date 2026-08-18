import streamlit as st
import numpy as np
import pickle

# Configure Streamlit page
st.set_page_config(
    page_title="Medical Condition Prediction",
    page_icon=":hospital:",
    layout="centered",
)

# -------------------------------
# Load Model and Scaler
# -------------------------------
with open("knn_model (1)", "rb") as f:
    model = pickle.load(f)

with open("scaler (1).pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------------------
# Streamlit App Title and Description
# -------------------------------
st.title("Medical Condition Prediction App")
st.markdown("Enter the patient's medical details to predict their condition.")

# -------------------------------
# Input Fields for Features
# -------------------------------
st.sidebar.header("Patient Data Input")

age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.sidebar.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
heart_rate = st.sidebar.number_input("Heart Rate", min_value=30, max_value=200, value=70)
systolic_bp = st.sidebar.number_input("Systolic Blood Pressure", min_value=70, max_value=250, value=120)
diastolic_bp = st.sidebar.number_input("Diastolic Blood Pressure", min_value=40, max_value=150, value=80)
blood_sugar = st.sidebar.number_input("Blood Sugar (mg/dL)", min_value=50, max_value=500, value=100)
ck_mb = st.sidebar.number_input("CK-MB (ng/mL)", min_value=0.0, max_value=100.0, value=1.0, format="%.2f")
troponin = st.sidebar.number_input("Troponin (ng/mL)", min_value=0.0, max_value=50.0, value=0.01, format="%.3f")

# -------------------------------
# Prediction Button and Logic
# -------------------------------
if st.sidebar.button("Predict"):
    input_data = np.array([
        age,
        gender,
        heart_rate,
        systolic_bp,
        diastolic_bp,
        blood_sugar,
        ck_mb,
        troponin
    ]).reshape(1, -1)

    # Scale the input data
    scaled_input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input_data)

    st.subheader("Prediction Result")
    if prediction[0] == 0:
        st.success("The model predicts: No Medical Condition")
    else:
        st.warning("The model predicts: Medical Condition Present")

st.markdown("---")
st.markdown("**Note:** This is a simplified model for demonstration purposes.")
