import gradio as gr
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
# Prediction Function
# ----------------------------
def predict_stress(ecg_mean, ecg_std, ecg_max, ecg_min,
                   eda_mean, eda_std,
                   temp_mean, temp_std,
                   hr_mean, hr_std):

    x = np.array([[ 
        ecg_mean, ecg_std, ecg_max, ecg_min,
        eda_mean, eda_std,
        temp_mean, temp_std,
        hr_mean, hr_std
    ]])

    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]

    # Probability (if available)
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(x_scaled)[0,1])
    else:
        prob = None

    label = "Stressed 😟" if pred == 1 else "Relaxed 🙂"

    return label, f"{prob*100:.2f}%" if prob is not None else "N/A"

# ----------------------------
# Gradio UI
# ----------------------------
inputs = [
    gr.Slider(-2.0, 2.0, 0.05, label="ECG Mean"),
    gr.Slider(0.0, 2.0, 0.1, label="ECG Std"),
    gr.Slider(-2.0, 2.0, 0.8, label="ECG Max"),
    gr.Slider(-2.0, 2.0, -0.8, label="ECG Min"),

    gr.Slider(0.0, 10.0, 0.5, label="EDA Mean"),
    gr.Slider(0.0, 5.0, 0.2, label="EDA Std"),

    gr.Slider(28.0, 38.0, 33.5, label="Temperature Mean (°C)"),
    gr.Slider(0.0, 1.0, 0.1, label="Temperature Std"),

    gr.Slider(40, 160, 75, label="Heart Rate Mean (bpm)"),
    gr.Slider(0.0, 20.0, 4.0, label="Heart Rate Std")
]

outputs = [
    gr.Textbox(label="Prediction"),
    gr.Textbox(label="Stress Probability")
]

demo = gr.Interface(
    fn=predict_stress,
    inputs=inputs,
    outputs=outputs,
    title="Stress Detection",
    description="Predict Stress vs Relaxed using physiological features.",
)

demo.launch()
