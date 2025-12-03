import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Churn Prediction", page_icon="🎯", layout="wide")

# Load model
@st.cache_resource
def load_model():
    if not os.path.exists('models/churn_model.pkl'):
        st.error("❌ Model not found! Please train the model first.")
        st.stop()
    model = joblib.load('models/churn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# Title
st.title("🎯 Customer Churn Prediction")
st.markdown("Predict customer churn using CSV upload or manual input")

# Tabs
tab1, tab2 = st.tabs(["📤 Upload CSV", "✍️ Manual Input"])

# ================== TAB 1: CSV Upload ==================
with tab1:
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        # Load data
        df = pd.read_csv(uploaded_file)
        original_df = df.copy()
        
        st.subheader("📊 Uploaded Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data preprocessing (same as training)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip().replace('', np.nan), errors='coerce')
            df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
        for c in df.select_dtypes(include=['object']).columns:
            df[c] = df[c].astype(str).str.strip()
        
        if 'customerID' in df.columns:
            customer_ids = df['customerID'].copy()
            df = df.drop(columns=['customerID'])
        else:
            customer_ids = None
        
        # Remove target if present
        has_churn = 'Churn' in df.columns
        if has_churn:
            actual_churn = df['Churn'].copy()
            df = df.drop(columns=['Churn'])
        
        # Clean service columns
        service_cols = [c for c in df.columns if any(s in c for s in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','MultipleLines'])]
        for c in service_cols:
            df[c] = df[c].replace({'No internet service':'No', 'No phone service':'No'})
        
        # Encode binaries
        binaries = [c for c in df.columns if df[c].dropna().isin(['Yes','No']).all()]
        for c in binaries:
            df[c] = df[c].map({'Yes':1, 'No':0})
        
        # One-hot encode categorical
        obj_cols = df.select_dtypes(include=['object']).columns.tolist()
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
        
        # Align features with training - add missing, keep only needed, same order
        for feat in feature_names:
            if feat not in df.columns:
                df[feat] = 0
        df = df[feature_names]
        
        # Scale only the columns the scaler was trained on
        if hasattr(scaler, 'feature_names_in_'):
            scale_cols = list(scaler.feature_names_in_)
            df[scale_cols] = scaler.transform(df[scale_cols])
        else:
            # Fallback: scale all numeric
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df[num_cols] = scaler.transform(df[num_cols])
        
        # Predict
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1] if hasattr(model, 'predict_proba') else predictions
        
        # Results
        st.subheader("🔮 Predictions")
        
        results = pd.DataFrame({
            'Customer ID': customer_ids if customer_ids is not None else range(len(predictions)),
            'Churn Prediction': ['Yes' if p == 1 else 'No' for p in predictions],
            'Churn Probability': [f"{prob:.1%}" for prob in probabilities]
        })
        
        if has_churn:
            results['Actual Churn'] = actual_churn
            results['Correct'] = ['✅' if pred == actual else '❌' for pred, actual in zip(predictions, actual_churn.map({'Yes':1, 'No':0}))]
        
        st.dataframe(results, use_container_width=True)
        
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", len(predictions))
        with col2:
            churn_count = sum(predictions)
            st.metric("Predicted Churns", churn_count)
        with col3:
            churn_rate = churn_count / len(predictions) * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        
        # Download results
        csv = results.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )
    else:
        st.info("👆 Please upload a CSV file to start predicting")

# ================== TAB 2: Manual Input ==================
with tab2:
    st.subheader("Enter Customer Details")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**👤 Demographics**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        
        with col2:
            st.markdown("**📞 Services**")
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        
        with col3:
            st.markdown("**📺 Streaming & Billing**")
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", 
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=monthly_charges * tenure, step=0.1)
        
        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
    
    if submitted:
        # Create dataframe from input
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })
        
        # Preprocess
        df = input_data.copy()
        
        # Clean service columns
        service_cols = [c for c in df.columns if any(s in c for s in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','MultipleLines'])]
        for c in service_cols:
            df[c] = df[c].replace({'No internet service':'No', 'No phone service':'No'})
        
        # Encode binaries
        binaries = [c for c in df.columns if df[c].dropna().isin(['Yes','No']).all()]
        for c in binaries:
            df[c] = df[c].map({'Yes':1, 'No':0})
        
        # One-hot encode categorical
        obj_cols = df.select_dtypes(include=['object']).columns.tolist()
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
        
        # Align features with training FIRST
        for feat in feature_names:
            if feat not in df.columns:
                df[feat] = 0
        
        # Keep only features from training (in same order)
        df = df[feature_names]
        
        # Scale only the columns the scaler was trained on
        if hasattr(scaler, 'feature_names_in_'):
            scale_cols = list(scaler.feature_names_in_)
            df[scale_cols] = scaler.transform(df[scale_cols])
        else:
            # Fallback: scale all numeric
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df[num_cols] = scaler.transform(df[num_cols])
        
        # Predict
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0, 1] if hasattr(model, 'predict_proba') else prediction
        
        # Display result
        st.markdown("---")
        st.subheader("🔮 Prediction Result")
        
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.error("### ⚠️ HIGH RISK - Customer Will Churn")
            else:
                st.success("### ✅ LOW RISK - Customer Will Stay")
        
        with col2:
            st.metric("Churn Probability", f"{probability:.1%}", 
                     delta=f"{abs(probability - 0.5):.1%} {'above' if probability > 0.5 else 'below'} average")
        
        # Recommendation
        st.markdown("---")
        st.subheader("💡 Recommendation")
        if probability > 0.7:
            st.warning("**Immediate Action Required:** Contact customer with retention offer")
        elif probability > 0.5:
            st.info("**Monitor Closely:** Customer shows signs of dissatisfaction")
        else:
            st.success("**Satisfied Customer:** Continue regular engagement")

st.markdown("---")
st.caption("💡 Model: Random Forest trained with SMOTE | Accuracy: ~79%")
