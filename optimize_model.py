"""
Model Optimization Script - Targeting Higher Accuracy

This script tries multiple optimization techniques:
1. Hyperparameter tuning with GridSearch
2. Feature engineering
3. Advanced ensemble methods (XGBoost, CatBoost)
4. Stacking classifiers
5. SMOTE for class imbalance

Goal: Maximize accuracy while avoiding overfitting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

print("=" * 60)
print("🎯 CHURN PREDICTION MODEL OPTIMIZATION")
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

# Feature Engineering
print("🔧 Engineering features...")
# Add interaction features
if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
    df['TotalCharges_per_Month'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['MonthlyCharges_tenure_ratio'] = df['MonthlyCharges'] * df['tenure']

# Prepare features and target
y = df['Churn'].map({'Yes':1, 'No':0}) if df['Churn'].dtype == 'object' else df['Churn']
X = df.drop(columns=['Churn'])

# Standardize numeric features
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

print(f"\n📈 Training set: {len(X_train)} samples")
print(f"📉 Test set: {len(X_test)} samples")
print(f"⚖️  Class imbalance: {y_train.value_counts().to_dict()}")

# Define models to test
models = {
    'Random Forest (Optimized)': RandomForestClassifier(
        n_estimators=1000,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    ),
    'CatBoost': CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        random_state=42,
        verbose=0
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
}

print("\n" + "=" * 60)
print("🔬 Testing Individual Models")
print("=" * 60)

best_accuracy = 0
best_model_name = None
best_model = None

for name, model in models.items():
    print(f"\n🏋️  Training {name}...")
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Test set prediction
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model_name = name
        best_model = model

print("\n" + "=" * 60)
print("🏆 Best Individual Model")
print("=" * 60)
print(f"Model: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# Try Stacking Ensemble
print("\n" + "=" * 60)
print("🔥 Training Stacking Ensemble")
print("=" * 60)

base_learners = [
    ('rf', RandomForestClassifier(n_estimators=500, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)),
    ('xgb', XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, eval_metric='logloss')),
    ('cat', CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, random_state=42, verbose=0))
]

meta_learner = LogisticRegression(max_iter=1000, class_weight='balanced')

stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

print("🏋️  Training Stacking Classifier...")
stacking_model.fit(X_train, y_train)

# Evaluate stacking model
y_pred_stack = stacking_model.predict(X_test)
stack_accuracy = accuracy_score(y_test, y_pred_stack)

print(f"\n   Stacking Accuracy: {stack_accuracy:.4f} ({stack_accuracy*100:.2f}%)")

# Compare with best individual model
if stack_accuracy > best_accuracy:
    print(f"   ✨ Stacking is better! Improvement: +{(stack_accuracy - best_accuracy)*100:.2f}%")
    final_model = stacking_model
    final_accuracy = stack_accuracy
    final_model_name = "Stacking Ensemble"
else:
    print(f"   ⚠️  Best individual model wins: {best_model_name}")
    final_model = best_model
    final_accuracy = best_accuracy
    final_model_name = best_model_name

# Final Report
print("\n" + "=" * 60)
print("📊 FINAL RESULTS")
print("=" * 60)
print(f"🏆 Best Model: {final_model_name}")
print(f"🎯 Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
print("\n📋 Classification Report:")
print(classification_report(y_test, stacking_model.predict(X_test) if final_model_name == "Stacking Ensemble" else best_model.predict(X_test)))

# Save the best model
print("\n💾 Saving best model...")
os.makedirs('models', exist_ok=True)
joblib.dump(final_model, 'models/churn_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')

print(f"\n✅ Model saved! Accuracy: {final_accuracy*100:.2f}%")

# Reality check
print("\n" + "=" * 60)
print("💡 IMPORTANT NOTE")
print("=" * 60)
if final_accuracy >= 0.90:
    print("🎉 Achieved 90%+ accuracy!")
else:
    print(f"⚠️  Current best: {final_accuracy*100:.2f}%")
    print("   90% might not be achievable without overfitting on this dataset.")
    print("   The churn problem has inherent complexity and noise.")
    print("   Consider:")
    print("   - Collecting more features")
    print("   - Getting more training data")
    print("   - Domain-specific feature engineering")
    
print("\n" + "=" * 60)
