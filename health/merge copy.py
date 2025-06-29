import pandas as pd

# Load files
diagnosis = pd.read_csv("/Users/nandhu/Downloads/diagnosis.csv")
edstays = pd.read_csv("/Users/nandhu/Downloads/edstays.csv")
triage = pd.read_csv("/Users/nandhu/Downloads/triage.csv")
vitalsign = pd.read_csv("/Users/nandhu/Downloads/vitalsign.csv")

# Merge on subject_id and stay_id
df = diagnosis.merge(edstays, on=['subject_id', 'stay_id'], how='left')
df = df.merge(triage, on=['subject_id', 'stay_id'], how='left')
df = df.merge(vitalsign, on=['subject_id', 'stay_id'], how='left')

# Drop unnecessary columns (e.g., icd_code, seq_num, time columns)
df.drop(columns=['seq_num', 'icd_code', 'icd_version', 'intime', 'outtime', 'temperature_site'], inplace=True, errors='ignore')

# Drop missing values (you can also choose to impute instead)
df.dropna(inplace=True)

# Show basic info
print("Shape after merge and clean:", df.shape)
print("ICD Labels:", df['icd_title'].value_counts().head(10))


print(df.columns.tolist())
import seaborn as sns
import matplotlib.pyplot as plt

# Plot class distribution
plt.figure(figsize=(10, 5))
df['icd_title'].value_counts().head(10).plot(kind='barh', color='skyblue')
plt.title("Top 10 ICD Diagnoses")
plt.xlabel("Number of Records")
plt.ylabel("ICD Title")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Corrected vital columns
vital_cols = ['heartrate_x', 'resprate_x', 'o2sat_x', 'sbp_x', 'dbp_x', 'temperature_x']

# Pairwise plot
sns.pairplot(df[vital_cols].dropna())
plt.suptitle("Pairwise Plot of Vital Signs", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[vital_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation of Vital Signs")
plt.show()

#label encoding
from sklearn.preprocessing import LabelEncoder

# Encode target variable
le = LabelEncoder()
df['icd_encoded'] = le.fit_transform(df['icd_title'])

# Save mapping for later use (e.g., Streamlit or inverse prediction)
icd_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("ICD Class Mapping:\n", icd_mapping)

#feature selection 
features = [
    'heartrate_x', 'resprate_x', 'o2sat_x', 'sbp_x', 'dbp_x', 'temperature_x',
    'gender', 'race', 'arrival_transport', 'disposition', 'acuity'
]

# One-hot encode categorical variables
df_model = pd.get_dummies(df[features], drop_first=True)

# Final dataset
X = df_model
y = df['icd_encoded']

# Filter classes with at least 2 records
valid_labels = df['icd_title'].value_counts()
valid_labels = valid_labels[valid_labels >= 2].index

# Keep only rows with these labels
df_filtered = df[df['icd_title'].isin(valid_labels)]

# Redefine X and y
X = df_filtered[['heartrate_x', 'resprate_x', 'o2sat_x', 'sbp_x', 'dbp_x', 'temperature_x']]
y = df_filtered['icd_title']

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

# Features and target
X = df[vital_cols]
y = df['icd_title']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier()
}

# Stratified K-Fold
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("Model Performance (Cross-Validation Accuracy):")
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y_encoded, cv=skf, scoring='accuracy')
    print(f"{name}: {scores.mean():.2f} ± {scores.std():.2f}")

df.to_csv("filtered_icd_dataset.csv", index=False)


# model_training.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load filtered dataset
df = pd.read_csv("filtered_icd_dataset.csv")

# Features and target
features = ['heartrate_x', 'sbp_x', 'dbp_x', 'resprate_x', 'o2sat_x', 'temperature_x']
X = df[features]
le = LabelEncoder()
y = le.fit_transform(df['icd_title'])

# Train final model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoder
joblib.dump(model, "final_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Model and label encoder saved.")




import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load preprocessed ICD dataset
df = pd.read_csv("/Users/nandhu/filtered_icd_dataset_1000.csv")

# Features and target
X = df[['heartrate_x', 'sbp_x', 'dbp_x', 'resprate_x', 'o2sat_x', 'temperature_x']]
y = df['icd_title']

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_scaled, y_encoded)

# Save components
joblib.dump(model, "final_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")







