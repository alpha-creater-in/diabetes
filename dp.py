# dp.py

import streamlit as st
import pandas as pd
import joblib

# ==========================================
# Page Config
# ==========================================
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# ==========================================
# Load Model
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

model = load_model()

# ==========================================
# Title
# ==========================================
st.title("🩺 Diabetes Risk Prediction System")
st.write("Enter patient details below to assess diabetes probability.")

# ==========================================
# Input Layout
# ==========================================
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100, 30)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    glucose = st.number_input("Glucose Level", 50, 300, 100)
    blood_pressure = st.number_input("Blood Pressure", 40, 200, 80)

with col2:
    cholesterol = st.number_input("Cholesterol", 100, 400, 180)
    insulin = st.number_input("Insulin", 0, 500, 80)
    family_history = st.selectbox("Family History of Diabetes", [0, 1])
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    sleep_hours = st.slider("Sleep Hours per Day", 1, 12, 7)

# ==========================================
# Prediction
# ==========================================
if st.button("Predict Risk"):

    input_data = pd.DataFrame([[
        age, bmi, glucose, blood_pressure,
        cholesterol, insulin,
        family_history, stress_level,
        sleep_hours
    ]], columns=[
        'age', 'bmi', 'glucose', 'blood_pressure',
        'cholesterol', 'insulin',
        'family_history', 'stress_level',
        'sleep_hours'
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Result")
    st.write(f"Diabetes Probability: {probability*100:.2f}%")
    st.write("Model Prediction:", "Diabetic" if prediction == 1 else "Non-Diabetic")

    if probability < 0.30:
        st.success("🟢 LOW RISK")
    elif probability < 0.70:
        st.warning("🟡 MEDIUM RISK")
    else:
        st.error("🔴 HIGH RISK")
