import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

print("🚀 Starting model training...")

# Load data
print("📊 Loading data...")
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Data cleaning
print("🧹 Cleaning data...")
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

# Prepare features and target
print("🎯 Preparing features and target...")
y = df['Churn'].map({'Yes':1, 'No':0}) if df['Churn'].dtype == 'object' else df['Churn']
X = df.drop(columns=['Churn'])

# Standardize numeric features
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])

# Train/test split
print("✂️ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Train model (using Random Forest - best model from notebook experiments)
print("🏋️ Training Random Forest model (best from experiments: ROC-AUC 0.840)...")
model = RandomForestClassifier(
    n_estimators=500,
    n_jobs=-1,
    random_state=50,
    max_features="sqrt",
    max_leaf_nodes=30,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluate
print("📈 Evaluating model...")
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"✅ Test Accuracy: {accuracy:.3f}")

# Save model and scaler
print("💾 Saving model and scaler...")
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/churn_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')

print("✨ Done! Model saved to 'models/churn_model.pkl'")
print(f"📊 Model accuracy: {accuracy:.1%}")
