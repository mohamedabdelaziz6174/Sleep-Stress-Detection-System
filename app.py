import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Stress Detection AI",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('SaYoPillow_Best_Model.pkl')
        scaler = joblib.load('SaYoPillow_Scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure .pkl files are in the directory.")
        return None, None

model, scaler = load_artifacts()

st.title("🧠 Human Stress Detection System")
st.markdown("### Physiological Sleep Data Analysis")
st.write("---")

col1, col2 = st.columns(2)

with col1:
    sr = st.number_input("Snoring Range (dB)", min_value=40.0, max_value=100.0, value=50.0, step=0.1)
    rr = st.number_input("Respiration Rate (bpm)", min_value=10.0, max_value=40.0, value=18.0, step=0.1)
    bt = st.number_input("Body Temperature (°F)", min_value=80.0, max_value=100.0, value=97.0, step=0.1)
    lm = st.number_input("Limb Movement Rate", min_value=0.0, max_value=30.0, value=10.0, step=0.1)

with col2:
    bo = st.number_input("Blood Oxygen (%)", min_value=70.0, max_value=100.0, value=95.0, step=0.1)
    rem = st.number_input("Eye Movement (REM)", min_value=50.0, max_value=120.0, value=90.0, step=0.1)
    sh = st.number_input("Sleeping Hours", min_value=0.0, max_value=12.0, value=7.0, step=0.1)
    hr = st.number_input("Heart Rate (bpm)", min_value=40.0, max_value=100.0, value=60.0, step=0.1)

st.write("---")

if st.button("Analyze Stress Level", use_container_width=True):
    if model:
        features = np.array([[sr, rr, bt, lm, bo, rem, sh, hr]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        result_labels = {
            0: ("Low / Normal", "success"),
            1: ("Medium Low", "info"),
            2: ("Medium", "warning"),
            3: ("Medium High", "warning"),
            4: ("High", "error")
        }

        label, status = result_labels.get(prediction, ("Unknown", "error"))

        st.metric(label="Predicted Stress Level", value=label)

        if status == "success":
            st.success("The patient is in a healthy, relaxed state.")
        elif status == "info":
            st.info("Mild stress detected. Regular sleep monitoring is advised.")
        elif status == "warning":
            st.warning("Elevated stress levels detected. Relaxation techniques recommended.")
        else:
            st.error("Critical stress level detected. Immediate medical attention is advised.")

st.markdown("---")
st.caption("AI-Powered Stress Detection System | Deployment v1.0")