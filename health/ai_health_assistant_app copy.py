import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import joblib
from xgboost import XGBRegressor
import mysql.connector
import plotly.graph_objects as go
import hashlib
from datetime import datetime

# ------------------ Setup ------------------
st.set_page_config(page_title="AI Health Assistant", layout="wide", page_icon="🏥")
st.markdown("""
    <style>
        .main {
            background-color: green;
        }
        .sidebar .sidebar-content {
            background-color: #e8f4f8;
        }
        .stButton>button, .stTextInput>div>input, .stSelectbox>div>div>div {
            background-color: #4a90e2;
            color: white;
            border-radius: 5px;
        }
        .stButton>button:hover, .stTextInput>div>input:hover, .stSelectbox>div>div>div:hover {
            background-color: #3a7bc8;
        }
        .css-1aumxhk {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏥 AI Health Assistant")
st.markdown("An intelligent assistant to **predict diseases**, estimate **insurance billing**, and analyze **reimbursement outcomes**.")

# ------------------ Database Functions ------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost", 
        user="root", 
        password="12345678", 
        database="health_insurance_ai"
    )

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def log_action(user_id, username, action):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_logs (user_id, username, action)
            VALUES (%s, %s, %s)
        """, (user_id, username, action))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Audit log error: {str(e)}")

# ------------------ Login Panel ------------------
st.sidebar.title("🔐 User Login")
login_container = st.sidebar.container()

with login_container:
    username_input = st.text_input("Username", key="username")
    password_input = st.text_input("Password", type="password", key="password")
    
    # Get available roles from database
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT role FROM users")
        roles = [role[0] for role in cursor.fetchall()]
        conn.close()
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        roles = ["provider", "coder", "admin", "billing"]
    
    role_input = st.selectbox("Select Your Role", roles, key="role")
    login_button = st.button("Login")

if login_button:
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM users 
            WHERE username=%s AND password_hash=%s AND role=%s
        """, (username_input, hash_password(password_input), role_input))
        user_info = cursor.fetchone()
        conn.close()

        if user_info:
            st.success(f"✅ Welcome, {user_info['username']}!")
            st.session_state['user'] = user_info
            log_action(user_info['user_id'], user_info['username'], "Logged in")
        else:
            st.error("❌ Invalid credentials or role")
    except Exception as e:
        st.error(f"Database error: {str(e)}")

if 'user' not in st.session_state:
    st.warning("⚠️ Please log in to continue.")
    st.stop()

user = st.session_state['user']
st.sidebar.success(f"Logged in as: {user['username']} ({user['role']})")

# ------------------ Main Navigation ------------------
options = ["🩺 Disease Diagnosis", "💸 Insurance Billing", "📥 Reimbursement", "📊 Dashboard"]
option = st.sidebar.radio("Navigation", options)

# Log navigation action
log_action(user['user_id'], user['username'], f"Navigated to {option}")
# ------------------ Disease Diagnosis Dashboard ------------------
if option == "🩺 Disease Diagnosis":
    st.header("🩺 Disease Prediction Dashboard")
    
    try:
        model = joblib.load("final_model.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        scaler = joblib.load("scaler.pkl")
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        st.stop()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Patient Vitals")
        hr = st.slider("❤️ Heart Rate (bpm)", 40, 180, 72)
        sbp = st.slider("💉 Systolic BP (mmHg)", 80, 200, 120)
        dbp = st.slider("💉 Diastolic BP (mmHg)", 50, 120, 80)
        resp = st.slider("🌬️ Respiration Rate", 10, 40, 18)
        o2 = st.slider("🫁 Oxygen Saturation (%)", 70, 100, 98)
        temp = st.slider("🌡️ Temperature (°F)", 95, 105, 98)
        
        try:
            user_input = pd.DataFrame([[hr, sbp, dbp, resp, o2, temp]],
                columns=['heartrate_x', 'sbp_x', 'dbp_x', 'resprate_x', 'o2sat_x', 'temperature_x'])

            user_input_scaled = scaler.transform(user_input)
            prediction = model.predict(user_input_scaled)
            disease = label_encoder.inverse_transform(prediction)[0]

            st.success(f"🔍 Predicted Disease: **{disease}**")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    
    with col2:
        st.subheader("Vitals Overview")
        vitals_data = {
            'Metric': ['Heart Rate', 'Systolic BP', 'Diastolic BP', 'Respiration', 'O2 Sat', 'Temperature'],
            'Value': [hr, sbp, dbp, resp, o2, temp],
            'Normal Range': ['60-100', '90-120', '60-80', '12-20', '95-100%', '97-99°F']
        }
        vitals_df = pd.DataFrame(vitals_data)
        st.dataframe(vitals_df, hide_index=True)
        
        # Visualize vitals
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(vitals_df['Metric'], vitals_df['Value'], color='skyblue')
        ax.set_title('Patient Vitals')
        ax.set_ylabel('Value')
        st.pyplot(fig)

# ------------------ Insurance Billing Dashboard ------------------
elif option == "💸 Insurance Billing":
    st.header("💸 Insurance Billing Dashboard")
    
    try:
        df = pd.read_csv("insurance_preprocessed.csv")
        X = df.drop(['charges'], axis=1)
        y = df['charges']

        reg_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        reg_model.fit(X, y)
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        name = st.text_input("👤 Patient Name", "John Doe")
        age = st.slider("🎂 Age", 18, 100, 30)
        bmi = st.slider("📏 BMI", 15.0, 50.0, 25.0)
        children = st.slider("👶 Children", 0, 5, 0)
    
    with col2:
        st.subheader("Health Factors")
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

        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💰 Estimated Charges", f"${predicted_charge:,.2f}")
        with col2:
            st.metric("📄 Claim Status", status)
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Minimum", "Average", "Maximum", "Your Prediction"],
            y=[y.min(), y.mean(), y.max(), predicted_charge],
            marker_color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'],
            text=[f"${y.min():,.0f}", f"${y.mean():,.0f}", f"${y.max():,.0f}", f"${predicted_charge:,.0f}"],
            textposition='auto'
        ))
        fig.update_layout(
            title="📉 Charges Comparison",
            xaxis_title="Category",
            yaxis_title="Amount ($)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Save to database
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO patients (name, age, sex, bmi, children, smoker, region, predicted_charge, claim_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (name, age, sex, bmi, children, smoker == 'yes', region, float(predicted_charge), status))
            conn.commit()
            conn.close()
            st.success("✅ Patient data saved to database")
        except Exception as e:
            st.error(f"Database error: {str(e)}")

# ------------------ Reimbursement Dashboard ------------------
elif option == "📥 Reimbursement":
    st.header("📥 Reimbursement Dashboard")
    
    try:
        df = pd.read_csv("insurance_reimbursement.csv")
        df_encoded = pd.get_dummies(df, columns=['insurance_provider', 'procedure_code'], drop_first=True)
        X = df_encoded.drop(['patient_id', 'actual_reimbursement', 'reimbursement_days', 'payment_status'], axis=1)
        y_reimbursement = df_encoded['actual_reimbursement']
        y_days = df_encoded['reimbursement_days']

        model_reimbursement = XGBRegressor(n_estimators=100, random_state=42)
        model_days = XGBRegressor(n_estimators=100, random_state=42)
        model_reimbursement.fit(X, y_reimbursement)
        model_days.fit(X, y_days)
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Claim Details")
        billed = st.slider("💰 Total Billed Amount", 1000, 100000, 20000)
        expected = st.slider("💸 Expected Reimbursement", 500, 90000, 15000)
    
    with col2:
        st.subheader("Service Details")
        provider = st.selectbox("🏥 Insurance Provider", df['insurance_provider'].unique())
        procedure = st.selectbox("🧾 Procedure Code", df['procedure_code'].unique())

    input_data = {'total_billed': [billed], 'expected_reimbursement': [expected]}
    for col in df_encoded.columns:
        if 'insurance_provider_' in col:
            input_data[col] = [1 if col.endswith(provider) else 0]
        elif 'procedure_code_' in col:
            input_data[col] = [1 if col.endswith(procedure) else 0]

    input_df = pd.DataFrame(input_data)
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X.columns]

    if st.button("📤 Predict Reimbursement"):
        predicted_amount = model_reimbursement.predict(input_df)[0]
        predicted_days = model_days.predict(input_df)[0]
        
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("✅ Predicted Reimbursement", f"${predicted_amount:,.2f}")
        with col2:
            st.metric("⏳ Estimated Delay", f"{int(predicted_days)} days")
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="number+gauge",
            value=predicted_amount,
            number={'prefix': "$"},
            domain={'x': [0, 1], 'y': [0.6, 1]},
            title={'text': "Reimbursement Amount"},
            gauge={
                'shape': "bullet",
                'axis': {'range': [0, billed]},
                'bar': {'color': "#3498db"},
                'steps': [
                    {'range': [0, billed*0.5], 'color': "#e74c3c"},
                    {'range': [billed*0.5, billed*0.8], 'color': "#f39c12"},
                    {'range': [billed*0.8, billed], 'color': "#2ecc71"}
                ]
            }
        ))
        fig.add_trace(go.Indicator(
            mode="number+gauge",
            value=predicted_days,
            number={'suffix': " days"},
            domain={'x': [0, 1], 'y': [0, 0.4]},
            title={'text': "Processing Time"},
            gauge={
                'shape': "bullet",
                'axis': {'range': [0, 60]},
                'bar': {'color': "#9b59b6"},
                'steps': [
                    {'range': [0, 15], 'color': "#2ecc71"},
                    {'range': [15, 30], 'color': "#f39c12"},
                    {'range': [30, 60], 'color': "#e74c3c"}
                ]
            }
        ))
        fig.update_layout(height=300, margin={'t': 0, 'b': 0, 'l': 0, 'r': 0})
        st.plotly_chart(fig, use_container_width=True)

# ------------------ Admin Dashboard ------------------
elif option == "📊 Dashboard":
    st.header("📊 System Dashboard")
    
    if user['role'] != 'admin':
        st.warning("⚠️ You need admin privileges to access this dashboard.")
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["📊 Statistics", "👥 User Management", "📋 Audit Logs"])
    
    with tab1:
        st.subheader("System Statistics")
        
        try:
            conn = get_db_connection()
            
            # Patient statistics
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM patients")
            patient_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(predicted_charge) FROM patients")
            avg_charge = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT claim_status, COUNT(*) 
                FROM patients 
                GROUP BY claim_status
            """)
            claim_stats = cursor.fetchall()
            
            st.metric("Total Patients", patient_count)
            st.metric("Average Charge", f"${avg_charge:,.2f}")
            
            if claim_stats:
                claim_df = pd.DataFrame(claim_stats, columns=['Status', 'Count'])
                fig = px.pie(claim_df, values='Count', names='Status', 
                            title='Claim Status Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            conn.close()
        except Exception as e:
            st.error(f"Database error: {str(e)}")
    
    with tab2:
        st.subheader("User Management")
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT user_id, username, role, created_at FROM users")
            users = cursor.fetchall()
            conn.close()
            
            if users:
                user_df = pd.DataFrame(users)
                st.dataframe(user_df)
            else:
                st.info("No users found in database")
        except Exception as e:
            st.error(f"Database error: {str(e)}")
    
    with tab3:
        st.subheader("Audit Trail")
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT username, action, timestamp 
                FROM audit_logs 
                ORDER BY timestamp DESC 
                LIMIT 100
            """)
            logs = cursor.fetchall()
            conn.close()
            
            if logs:
                audit_df = pd.DataFrame(logs)
                st.dataframe(audit_df)
                
                # Visualize actions
                action_counts = audit_df['action'].value_counts().reset_index()
                action_counts.columns = ['Action', 'Count']
                
                fig = px.bar(action_counts, x='Action', y='Count', 
                            title='Recent Activity Breakdown')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No audit logs found")
        except Exception as e:
            st.error(f"Database error: {str(e)}")

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("© 2025 AI Health Assistant | [Privacy Policy]() | [Terms of Service]()")
