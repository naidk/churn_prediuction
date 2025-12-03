"""
Training script with SMOTE (Balanced Data)
This version balances the classes before training to potentially improve accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

print("=" * 60)
print("🎯 TRAINING WITH BALANCED DATA (SMOTE)")
print("=" * 60)

# Load data
print("\n📊 Loading data...")
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

# Scale BEFORE split (for SMOTE)
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])

# Train/test split
print("✂️ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

print(f"\n📊 BEFORE SMOTE:")
print(f"   Training samples: {len(X_train)}")
print(f"   Class distribution: {dict(y_train.value_counts())}")

# Apply SMOTE to balance classes
print("\n⚖️  Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"\n📊 AFTER SMOTE:")
print(f"   Training samples: {len(X_train_balanced)}")
print(f"   Class distribution: {dict(pd.Series(y_train_balanced).value_counts())}")

# Train model (Random Forest - best from experiments)
print("\n🏋️  Training Random Forest on BALANCED data...")
model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_balanced, y_train_balanced)

# Evaluate
print("\n📈 Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

print("\n🔍 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"   True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"   False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

# Save model
print("\n💾 Saving model...")
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/churn_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')

print(f"\n✨ Done! Model saved to 'models/churn_model.pkl'")
print(f"📊 Final accuracy: {accuracy*100:.2f}%")

# Summary
print("\n" + "=" * 60)
print("📝 SUMMARY")
print("=" * 60)
if accuracy >= 0.90:
    print("🎉 SUCCESS! Achieved 90%+ accuracy with balanced data!")
elif accuracy >= 0.85:
    print(f"👍 Good! {accuracy*100:.2f}% accuracy - close to target!")
    print("   Consider if this is acceptable for your use case.")
else:
    print(f"📊 Achieved {accuracy*100:.2f}% accuracy")
    print("   SMOTE helped but 90% remains challenging for this dataset.")
    print("   This is normal - churn prediction is inherently difficult!")

print("\n💡 Remember: Higher accuracy on balanced data may mean")
print("   the model predicts MORE churns (higher recall).")
print("=" * 60)
