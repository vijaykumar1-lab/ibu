# Import libraries
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report

# -------------------------------
# Load preprocessed dataset
# -------------------------------
df = pd.read_csv("insurance_preprocessed.csv")

# -------------------------------
# Part 1: Healthcare Billing Prediction (Regression)
# -------------------------------

# Features and target
X = df.drop(['charges'], axis=1)
y = df['charges']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Regression model
xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_test)

# Evaluation
print("\n--- XGBoost Billing Prediction ---")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save billing predictions
df['predicted_charges'] = xgb_reg.predict(X)

# -------------------------------
# Part 2: Claim Acceptance Prediction (Classification)
# -------------------------------

# Create binary label: accepted = 1 if charges < 20000
df['accepted'] = df['charges'].apply(lambda x: 1 if x < 20000 else 0)

# Classification features & labels
X2 = df.drop(['charges', 'accepted', 'predicted_charges'], axis=1)
y2 = df['accepted']

# Train/Test Split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# XGBoost Classifier
xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_clf.fit(X2_train, y2_train)
y2_pred = xgb_clf.predict(X2_test)

# Evaluation
print("\n--- XGBoost Claim Acceptance Prediction ---")
print("Accuracy:", accuracy_score(y2_test, y2_pred))
print(classification_report(y2_test, y2_pred))

# Save acceptance predictions
df['predicted_accepted'] = xgb_clf.predict(X2)

# -------------------------------
# Save Final Output
# -------------------------------
df.to_csv("xgboost_predictions_final.csv", index=False)
print("\n✅ Predictions saved to 'xgboost_predictions_final.csv'")