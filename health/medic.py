# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load dataset (you can replace with a medical dataset later)
df = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

# -------------------------------
# 1. Inspect the data
# -------------------------------
print("Head of dataset:\n", df.head())
print("\nInfo:\n")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

# -------------------------------
# 2. Handle Missing Values (if any)
# -------------------------------
df.fillna(method='ffill', inplace=True)

# -------------------------------
# 3. Feature Engineering
# -------------------------------
# BMI groups
df['bmi_group'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], 
                         labels=["Underweight", "Normal", "Overweight", "Obese"])
df['bmi_group']
# -------------------------------
# 4. Encoding Categorical Variables
# -------------------------------
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region', 'bmi_group'], drop_first=True)
df_encoded
# -------------------------------
# 5. Feature Scaling (Normalization)
# -------------------------------
scaler = StandardScaler()
df_encoded[['age', 'bmi', 'children']] = scaler.fit_transform(df_encoded[['age', 'bmi', 'children']])
scaler
# -------------------------------
# 6. Visualization
# -------------------------------

# Histogram of charges
plt.figure(figsize=(8, 5))
sns.histplot(df['charges'], kde=True)
plt.title("Distribution of Charges")
plt.xlabel("Charges")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Charges by smoker
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='smoker', y='charges')
plt.title("Charges by Smoking Status")
plt.grid(True)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df, hue='smoker', vars=['age', 'bmi', 'charges'])
plt.suptitle("Pairplot by Smoker", y=1.02)
plt.show()

from sklearn.model_selection import train_test_split

X = df_encoded.drop('charges', axis=1)  # Features
y = df_encoded['charges']               # Target (Healthcare Billing Charges)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Healthcare Charges")
plt.grid(True)
plt.show()

#xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Initialize XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Fit the model
xgb_model.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb_model.predict(X_test)

# Evaluation
print("MAE (XGBoost):", mean_absolute_error(y_test, y_pred_xgb))
print("R2 Score (XGBoost):", r2_score(y_test, y_pred_xgb))

# From earlier
print("MAE (Random Forest):", mean_absolute_error(y_test, y_pred))
print("R2 Score (Random Forest):", r2_score(y_test, y_pred))

# Now with XGBoost
print("MAE (XGBoost):", mean_absolute_error(y_test, y_pred_xgb))
print("R2 Score (XGBoost):", r2_score(y_test, y_pred_xgb))

# Save to local CSV
df_encoded.to_csv("insurance_preprocessed.csv", index=False)


import pandas as pd

df = pd.read_csv("insurance_preprocessed.csv")

# Create binary classification label: claim accepted (1 if charges < 20000)
df['accepted'] = df['charges'].apply(lambda x: 1 if x < 20000 else 0)
X2 = df.drop(['charges', 'accepted'], axis=1)  # Features
y2 = df['accepted']                            # Label
from sklearn.model_selection import train_test_split

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
#train and evaluate both models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X2_train, y2_train)
rf_preds = rf_model.predict(X2_test)

# XGBoost
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X2_train, y2_train)
xgb_preds = xgb_model.predict(X2_test)

# Evaluation
print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y2_test, rf_preds))
print(classification_report(y2_test, rf_preds))

print("\n--- XGBoost ---")
print("Accuracy:", accuracy_score(y2_test, xgb_preds))
print(classification_report(y2_test, xgb_preds))

# Conclusion
if accuracy_score(y2_test, rf_preds) > accuracy_score(y2_test, xgb_preds):
    print("\n✅ Random Forest performed better.")
else:
    print("\n✅ XGBoost performed better.")





























