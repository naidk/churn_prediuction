"""Quick script to check model accuracy"""
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load model
model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Load and prepare test data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df.copy()
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip().replace('', np.nan), errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].astype(str).str.strip()

if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

service_cols = [c for c in df.columns if any(s in c for s in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','MultipleLines'])]
for c in service_cols:
    df[c] = df[c].replace({'No internet service':'No', 'No phone service':'No'})

binaries = [c for c in df.columns if df[c].dropna().isin(['Yes','No']).all()]
for c in binaries:
    df[c] = df[c].map({'Yes':1, 'No':0})

obj_cols = [c for c in df.select_dtypes(include=['object']).columns if c != 'Churn']
df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

y = df['Churn'].map({'Yes':1, 'No':0}) if df['Churn'].dtype == 'object' else df['Churn']
X = df.drop(columns=['Churn'])

# Prepare data same way as training
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
X[num_features] = scaler.transform(X[num_features])
X = X[feature_names]

# Split same way
_, X_test, _, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Predict
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()

print(f"\nModel Type: {type(model).__name__}")
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Trained with SMOTE: YES")
