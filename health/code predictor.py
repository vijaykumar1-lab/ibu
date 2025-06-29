# icd_streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load model and label encoder
model = joblib.load("final_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Streamlit UI
st.set_page_config(page_title="ICD Disease Predictor", layout="centered")
st.title("🩺 Disease Prediction from Vitals")

# Sidebar inputs
st.sidebar.header("Patient Vitals")
hr = st.sidebar.slider("Heart Rate (bpm)", 40, 180, 72)
sbp = st.sidebar.slider("Systolic BP (mmHg)", 80, 200, 120)
dbp = st.sidebar.slider("Diastolic BP (mmHg)", 50, 120, 80)
resp = st.sidebar.slider("Respiration Rate", 10, 40, 18)
o2 = st.sidebar.slider("Oxygen Saturation (%)", 70, 100, 98)
temp = st.sidebar.slider("Temperature (°F)", 95, 105, 98)

# Prepare input
user_input = pd.DataFrame([[hr, sbp, dbp, resp, o2, temp]],
    columns=['heartrate_x', 'sbp_x', 'dbp_x', 'resprate_x', 'o2sat_x', 'temperature_x'])

# Predict
prediction = model.predict(user_input)
disease = label_encoder.inverse_transform(prediction)[0]
st.success(f"🔍 Predicted Disease: **{disease}**")

# Comparison chart
st.subheader("📊 Vitals Comparison with Normal Ranges")

normal_values = {
    'heartrate_x': 72,
    'sbp_x': 120,
    'dbp_x': 80,
    'resprate_x': 18,
    'o2sat_x': 98,
    'temperature_x': 98.6
}

labels = list(normal_values.keys())
normal = list(normal_values.values())
user = list(user_input.iloc[0].values)
x = range(len(labels))

fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.35
ax.bar(x, normal, bar_width, label='Normal', color='#4CAF50')
ax.bar([p + bar_width for p in x], user, bar_width, label='You', color='#FF5722')
ax.set_xticks([p + bar_width/2 for p in x])
ax.set_xticklabels(labels, rotation=45)
ax.set_ylabel('Measurement')
ax.set_title('Your Vitals vs Normal Ranges')
ax.legend()
st.pyplot(fig)