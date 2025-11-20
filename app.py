import streamlit as st
import numpy as np
import pickle

# ----------------------------
# Load Model & Scaler
# ----------------------------
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ----------------------------
# Streamlit UI Setup
# ----------------------------
st.set_page_config(page_title="Stress Detection", layout="centered")
st.title("🧠 Stress Detection from Wearable Sensor Features")

st.write("""
Adjust the physiological feature values using the sliders below
to classify a sample as **Relaxed** or **Stressed**.
""")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("ℹ️ About the Model")
st.sidebar.write("""
This classifier uses features extracted from:
- ECG (electrocardiogram)
- EDA (electrodermal activity)
- Skin Temperature
- Heart Rate

The model used: **Best Performing Classifier**
""")

# ----------------------------
# Sliders for 10 Features
# ----------------------------

st.subheader("ECG Features")
ecg_mean = st.slider("ECG Mean", -2.0, 2.0, 0.05)
ecg_std = st.slider("ECG Std", 0.0, 2.0, 0.1)
ecg_max = st.slider("ECG Max", -2.0, 2.0, 0.8)
ecg_min = st.slider("ECG Min", -2.0, 2.0, -0.8)

st.subheader("EDA Features")
eda_mean = st.slider("EDA Mean", 0.0, 10.0, 0.5)
eda_std = st.slider("EDA Std", 0.0, 5.0, 0.2)

st.subheader("Temperature Features")
temp_mean = st.slider("Temperature Mean (°C)", 28.0, 38.0, 33.5)
temp_std = st.slider("Temperature Std", 0.0, 1.0, 0.1)

st.subheader("Heart Rate Features")
hr_mean = st.slider("Heart Rate Mean (bpm)", 40, 160, 75)
hr_std = st.slider("Heart Rate Std", 0.0, 20.0, 4.0)

# ----------------------------
# Prepare Input
# ----------------------------
input_features = np.array([[ 
    ecg_mean, ecg_std, ecg_max, ecg_min,
    eda_mean, eda_std,
    temp_mean, temp_std,
    hr_mean, hr_std
]])

scaled = scaler.transform(input_features)

# ----------------------------
# Predict
# ----------------------------
if st.button("🔍 Predict Stress Level"):
    pred = model.predict(scaled)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(scaled)[0,1]
    else:
        prob = 0

    st.write("### Result:")
    if pred == 1:
        st.error("⚠️ **STRESSED**")
    else:
        st.success("🟢 **RELAXED**")

    st.write(f"### Stress Probability: **{prob*100:.2f}%**")

# Footer
st.write("---")
st.write("Developed for AIML Mini Project — WESAD Stress Classification")
