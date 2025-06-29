import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from xgboost import XGBRegressor
import plotly.graph_objects as go

# ----------------- Setup -----------------
st.set_page_config(page_title="AI Health Assistant", layout="centered", page_icon="🤖")
st.title("🤖 AI Health Assistant")
st.markdown("An intelligent assistant to **predict diseases** from vitals and estimate **insurance billing** & claim outcomes.")

st.sidebar.title("🔘 Choose Task")
option = st.sidebar.radio("Select what you want to do:", ["🩺 Disease Diagnosis", "💸 Insurance Billing & Claim"])

# ----------------- Disease Diagnosis -----------------
if option == "🩺 Disease Diagnosis":
    st.header("🩺 Disease Prediction from Patient Vitals")

    # Load model and encoder
    model = joblib.load("final_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    # Sidebar inputs
    st.sidebar.subheader("Patient Vitals")
    hr = st.sidebar.slider("❤️ Heart Rate (bpm)", 40, 180, 72)
    sbp = st.sidebar.slider("💉 Systolic BP (mmHg)", 80, 200, 120)
    dbp = st.sidebar.slider("💉 Diastolic BP (mmHg)", 50, 120, 80)
    resp = st.sidebar.slider("🌬️ Respiration Rate", 10, 40, 18)
    o2 = st.sidebar.slider("🫁 Oxygen Saturation (%)", 70, 100, 98)
    temp = st.sidebar.slider("🌡️ Temperature (°F)", 95, 105, 98)

    user_input = pd.DataFrame([[hr, sbp, dbp, resp, o2, temp]],
        columns=['heartrate_x', 'sbp_x', 'dbp_x', 'resprate_x', 'o2sat_x', 'temperature_x'])

    prediction = model.predict(user_input)
    disease = label_encoder.inverse_transform(prediction)[0]

    st.success(f"🔍 Predicted Disease: **{disease}**")

    # Comparison chart
    st.subheader("📊 Vitals Comparison")

    normal_values = {
        'heartrate_x': 72,
        'sbp_x': 120,
        'dbp_x': 80,
        'resprate_x': 18,
        'o2sat_x': 98,
        'temperature_x': 98.6
    }

    labels = ['Heart Rate', 'Systolic BP', 'Diastolic BP', 'Respiration', 'O2 Saturation', 'Temperature']
    normal = list(normal_values.values())
    user = list(user_input.iloc[0].values)

    x = range(len(labels))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, normal, width=bar_width, label='Normal', color='lightgreen')
    ax.bar([i + bar_width for i in x], user, width=bar_width, label='User', color='skyblue')

    ax.set_xticks([i + bar_width / 2 for i in x])
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Value")
    ax.set_title("Vitals vs Normal Ranges")
    ax.legend()

    st.pyplot(fig)

# ----------------- Insurance Billing -----------------
elif option == "💸 Insurance Billing & Claim":
    st.header("💸 Insurance Billing & Claim Estimator")

    # Load and train model
    df = pd.read_csv("insurance_preprocessed.csv")
    X = df.drop(['charges'], axis=1)
    y = df['charges']

    reg_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    reg_model.fit(X, y)

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("🎂 Age", 18, 100, 30)
        bmi = st.slider("📏 BMI", 15.0, 50.0, 25.0)
        children = st.slider("👶 Children", 0, 5, 0)
    with col2:
        sex = st.selectbox("🧑 Sex", ["female", "male"])
        smoker = st.selectbox("🚬 Smoker?", ["no", "yes"])
        region = st.selectbox("📍 Region", ["northeast", "northwest", "southeast", "southwest"])

    bmi_group = (
        "underweight" if bmi < 18.5 else
        "Normal" if bmi < 25 else
        "Overweight" if bmi < 30 else
        "Obese"
    )

    input_dict = {
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex_male': [sex == 'male'],
        'smoker_yes': [smoker == 'yes'],
        'region_northwest': [region == 'northwest'],
        'region_southeast': [region == 'southeast'],
        'region_southwest': [region == 'southwest'],
        'bmi_group_Normal': [bmi_group == "Normal"],
        'bmi_group_Overweight': [bmi_group == "Overweight"],
        'bmi_group_Obese': [bmi_group == "Obese"]
    }

    input_df = pd.DataFrame(input_dict)

    if st.button("🔮 Predict Charges"):
        predicted_charge = reg_model.predict(input_df)[0]
        accepted = predicted_charge < 20000
        status = "✅ Likely Accepted" if accepted else "❌ Likely Rejected"

        st.subheader("📊 Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💰 Estimated Charges", f"${predicted_charge:,.2f}")
        with col2:
            st.metric("📄 Claim Status", status)

        avg_charge = y.mean()
        min_charge = y.min()
        max_charge = y.max()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=["Minimum", "Average", "Maximum", "Your Prediction"],
            y=[min_charge, avg_charge, max_charge, predicted_charge],
            mode='lines+markers',
            name='Comparison',
            line=dict(color='royalblue', width=3),
            marker=dict(size=10)
        ))
        fig.add_trace(go.Scatter(
            x=["Your Prediction"],
            y=[predicted_charge],
            mode='markers+text',
            name='Your Prediction',
            marker=dict(color='red', size=12),
            text=[f"${predicted_charge:,.2f}"],
            textposition="top center"
        ))

        fig.update_layout(
            title="📉 Charges Compared to Reference Values",
            xaxis_title="Reference",
            yaxis_title="Charges ($)",
            template="plotly_white",
            height=500,
            font=dict(size=14)
        )
        st.plotly_chart(fig, use_container_width=True)
