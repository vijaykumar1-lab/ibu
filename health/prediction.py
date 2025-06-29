import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Load data and train model
df = pd.read_csv("insurance_preprocessed.csv")
X = df.drop(['charges'], axis=1)
y = df['charges']

reg_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
reg_model.fit(X, y)

# Streamlit UI
st.set_page_config(page_title="Insurance Predictor", layout="centered", page_icon="🏥")
st.title("🏥 Insurance Billing & Claim Acceptance")
st.markdown("Use the form below to predict your health insurance charges and whether the claim might be accepted.")

# Input form in columns
col1, col2 = st.columns(2)
with col1:
    age = st.slider("🎂 Age", 18, 100, 30)
    bmi = st.slider("📏 BMI", 15.0, 50.0, 25.0)
    children = st.slider("👶 Number of Children", 0, 5, 0)
with col2:
    sex = st.selectbox("🧑 Sex", ["female", "male"])
    smoker = st.selectbox("🚬 Smoker?", ["no", "yes"])
    region = st.selectbox("📍 Region", ["northeast", "northwest", "southeast", "southwest"])

# BMI Group logic
if bmi < 18.5:
    bmi_group = "underweight"
elif bmi < 25:
    bmi_group = "Normal"
elif bmi < 30:
    bmi_group = "Overweight"
else:
    bmi_group = "Obese"

# Prepare input
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

# Predict and display
if st.button("🔮 Predict"):
    predicted_charge = reg_model.predict(input_df)[0]
    accepted = predicted_charge < 20000
    status = "✅ Likely Accepted" if accepted else "❌ Likely Rejected"

    st.subheader("📊 Prediction Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💰 Estimated Charges", f"${predicted_charge:,.2f}")
    with col2:
        st.metric("📄 Claim Status", status)

    # 📊 Interactive Plotly Chart: Predicted vs Min, Avg, Max
    import plotly.graph_objects as go

    avg_charge = y.mean()
    min_charge = y.min()
    max_charge = y.max()

    fig = go.Figure()

    # Comparison line
    fig.add_trace(go.Scatter(
        x=["Minimum", "Average", "Maximum", "Your Prediction"],
        y=[min_charge, avg_charge, max_charge, predicted_charge],
        mode='lines+markers',
        name='Comparison',
        line=dict(color='royalblue', width=3),
        marker=dict(size=10)
    ))

    # Highlight predicted value
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
        title="📉 Your Charges Compared to Reference Values",
        xaxis_title="Reference",
        yaxis_title="Charges ($)",
        template="plotly_dark",
        height=500,
        font=dict(size=14)
    )

    st.plotly_chart(fig, use_container_width=True)