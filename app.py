
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

MODEL_PATH = Path(__file__).parent / "diabetes_prediction_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("Diabetes Prediction Using Machine Learning")
st.write("Enter the patient details to get a model prediction.")

st.warning("Educational project only. This is not a medical diagnosis.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

with st.form("diabetes_form"):
    pregnancies = st.number_input("Pregnancies", 0, 20, 2)
    glucose = st.number_input("Glucose", 0, 300, 120)
    blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 25)
    insulin = st.number_input("Insulin", 0, 1000, 80)
    bmi = st.number_input("BMI", 0.0, 100.0, 28.5)
    pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.45)
    age = st.number_input("Age", 1, 120, 30)

    submit = st.form_submit_button("Predict")

if submit:
    patient = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age
    }])

    try:
        prediction = int(model.predict(patient)[0])

        if prediction == 1:
            st.error("Model Prediction: Diabetes-positive")
        else:
            st.success("Model Prediction: Diabetes-negative")

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(patient)[0][1]
            st.metric("Estimated Class 1 Probability", f"{probability * 100:.2f}%")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
