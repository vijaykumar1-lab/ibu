from flask import Flask, render_template_string, request, redirect, session, url_for
import mysql.connector
import pandas as pd
import plotly.graph_objs as go
import joblib
import hashlib
from xgboost import XGBRegressor
import numpy as np

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# ------------------ Utilities ------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345678",
        database="health_insurance_ai"
    )

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
        print(f"Log Error: {e}")

# ------------------ Login Page ------------------
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])
        role = request.form['role']

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username=%s AND password_hash=%s AND role=%s",
                       (username, password, role))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['user'] = user
            log_action(user['user_id'], user['username'], "Logged in")
            return redirect(url_for('dashboard'))
        else:
            return "Invalid credentials"

    # HTML form for login
    html = '''
    <h2>Login</h2>
    <form method="post">
        Username: <input type="text" name="username"><br>
        Password: <input type="password" name="password"><br>
        Role: <select name="role">
            <option value="provider">Provider</option>
            <option value="coder">Coder</option>
            <option value="admin">Admin</option>
            <option value="billing">Billing</option>
        </select><br>
        <input type="submit" value="Login">
    </form>
    '''
    return render_template_string(html)

# ------------------ Dashboard ------------------
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session:
        return redirect('/')

    user = session['user']

    html = '''
    <h2>Welcome {{ user['username'] }} ({{ user['role'] }})</h2>
    <a href="/predict">Disease Prediction</a><br>
    <a href="/billing">Insurance Billing</a><br>
    <a href="/reimbursement">Reimbursement Forecast</a><br>
    {% if user['role'] == 'admin' %}<a href="/logs">Audit Logs</a><br>{% endif %}
    <a href="/logout">Logout</a>
    '''
    return render_template_string(html, user=user)

# ------------------ Disease Prediction ------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    model = joblib.load("final_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("label_encoder.pkl")

    disease = None
    fig_html = ""

    if request.method == 'POST':
        hr = float(request.form['hr'])
        sbp = float(request.form['sbp'])
        dbp = float(request.form['dbp'])
        resp = float(request.form['resp'])
        o2 = float(request.form['o2'])
        temp = float(request.form['temp'])

        X = pd.DataFrame([[hr, sbp, dbp, resp, o2, temp]],
                         columns=['heartrate_x', 'sbp_x', 'dbp_x', 'resprate_x', 'o2sat_x', 'temperature_x'])
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)
        disease = encoder.inverse_transform(prediction)[0]

        trace1 = go.Bar(x=['HR', 'SBP', 'DBP', 'Resp', 'O2', 'Temp'],
                       y=[72, 120, 80, 18, 98, 98.6], name='Normal')
        trace2 = go.Bar(x=['HR', 'SBP', 'DBP', 'Resp', 'O2', 'Temp'],
                       y=[hr, sbp, dbp, resp, o2, temp], name='User')

        fig = go.Figure(data=[trace1, trace2])
        fig.update_layout(title='Vitals Comparison', barmode='group')
        fig_html = fig.to_html(full_html=False)

    return render_template_string('''
    <h2>Disease Prediction</h2>
    <form method="post">
        HR: <input type="number" name="hr" step="0.1" required><br>
        SBP: <input type="number" name="sbp" step="0.1" required><br>
        DBP: <input type="number" name="dbp" step="0.1" required><br>
        Resp: <input type="number" name="resp" step="0.1" required><br>
        O2 Sat: <input type="number" name="o2" step="0.1" required><br>
        Temp: <input type="number" name="temp" step="0.1" required><br>
        <input type="submit" value="Predict">
    </form>
    {% if disease %}<h3>Predicted: {{ disease }}</h3>{% endif %}
    {{ fig_html | safe }}
    <a href="/dashboard">Back</a>
    ''', disease=disease, fig_html=fig_html)

# ------------------ Insurance Billing ------------------
@app.route('/billing', methods=['GET', 'POST'])
def billing():
    df = pd.read_csv("insurance_preprocessed.csv")
    X = df.drop(['charges'], axis=1)
    y = df['charges']
    model = XGBRegressor()
    model.fit(X, y)
    result = ""

    if request.method == 'POST':
        age = int(request.form['age'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        sex = request.form['sex']
        smoker = request.form['smoker']
        region = request.form['region']

        bmi_group = "underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"

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
        predicted_charge = model.predict(input_df)[0]
        result = f"Estimated Charge: ${predicted_charge:,.2f}"

    return render_template_string('''
    <h2>Insurance Billing</h2>
    <form method="post">
        Age: <input type="number" name="age"><br>
        BMI: <input type="number" step="0.1" name="bmi"><br>
        Children: <input type="number" name="children"><br>
        Sex: <select name="sex"><option value="female">Female</option><option value="male">Male</option></select><br>
        Smoker: <select name="smoker"><option value="no">No</option><option value="yes">Yes</option></select><br>
        Region: <select name="region"><option value="northeast">Northeast</option><option value="northwest">Northwest</option><option value="southeast">Southeast</option><option value="southwest">Southwest</option></select><br>
        <input type="submit" value="Estimate">
    </form>
    <h3>{{ result }}</h3>
    <a href="/dashboard">Back</a>
    ''', result=result)

# ------------------ Reimbursement ------------------
@app.route('/reimbursement', methods=['GET', 'POST'])
def reimbursement():
    df = pd.read_csv("insurance_reimbursement.csv")
    df_encoded = pd.get_dummies(df, columns=['insurance_provider', 'procedure_code'], drop_first=True)
    X = df_encoded.drop(['patient_id', 'actual_reimbursement', 'reimbursement_days', 'payment_status'], axis=1)
    y_amount = df_encoded['actual_reimbursement']
    y_days = df_encoded['reimbursement_days']

    model_amount = XGBRegressor()
    model_days = XGBRegressor()
    model_amount.fit(X, y_amount)
    model_days.fit(X, y_days)

    result_amount = ""
    result_days = ""

    if request.method == 'POST':
        billed = float(request.form['billed'])
        expected = float(request.form['expected'])
        provider = request.form['provider']
        procedure = request.form['procedure']

        input_data = {'total_billed': [billed], 'expected_reimbursement': [expected]}

        for col in X.columns:
            if col.startswith('insurance_provider_'):
                input_data[col] = [1 if col.endswith(provider) else 0]
            elif col.startswith('procedure_code_'):
                input_data[col] = [1 if col.endswith(procedure) else 0]

        input_df = pd.DataFrame(input_data)
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[X.columns]

        amt = model_amount.predict(input_df)[0]
        days = model_days.predict(input_df)[0]
        result_amount = f"Predicted Reimbursement: ${amt:,.2f}"
        result_days = f"Expected Delay: {int(days)} days"

    return render_template_string('''
    <h2>Reimbursement Estimator</h2>
    <form method="post">
        Billed Amount: <input type="number" step="0.01" name="billed"><br>
        Expected Reimbursement: <input type="number" step="0.01" name="expected"><br>
        Provider: <input type="text" name="provider"><br>
        Procedure Code: <input type="text" name="procedure"><br>
        <input type="submit" value="Estimate">
    </form>
    <h3>{{ result_amount }}</h3>
    <h3>{{ result_days }}</h3>
    <a href="/dashboard">Back</a>
    ''', result_amount=result_amount, result_days=result_days)

# ------------------ Audit Logs ------------------
@app.route('/logs')
def logs():
    if 'user' not in session or session['user']['role'] != 'admin':
        return redirect('/')
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username, action, timestamp FROM audit_logs ORDER BY timestamp DESC LIMIT 100")
    logs = cursor.fetchall()
    conn.close()

    html = "<h2>Audit Logs</h2><table border=1><tr><th>Username</th><th>Action</th><th>Timestamp</th></tr>"
    for row in logs:
        html += f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td></tr>"
    html += "</table><a href='/dashboard'>Back</a>"
    return html

# ------------------ Logout ------------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# ------------------ Run App ------------------
if __name__ == '__main__':
    app.run(debug=True)
