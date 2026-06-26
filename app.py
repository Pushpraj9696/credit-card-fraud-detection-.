import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Page Configuration
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

st.title("💳 Credit Card Fraud Detection Web App")
st.write("This app uses a Machine Learning model to classify credit card transactions as Genuine or Fraudulent.")

# Mock/Cached Data Loader & Trainer for demonstration purpose
@st.cache_resource
def train_mock_model():
    # Creating simulated data to mimic the dataset from 'Screenshot_20260626-180432_Drive.png'
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 5) # 5 Principal Components (V1, V2, V3, V4, V5)
    X = np.hstack((X, np.random.exponential(scale=100, size=(n_samples, 1)))) # Amount
    
    y = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02]) # Class imbalance
    
    # Preprocess
    scaler = StandardScaler()
    X[:, -1] = scaler.fit_transform(X[:, -1].reshape(-1, 1)) # Scale Amount
    
    # Handle Class Imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Train Model
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_res, y_res)
    
    return model, scaler

model, scaler = train_mock_model()

# --- Sidebar Inputs for User ---
st.sidebar.header("📥 Input Transaction Details")

v1 = st.sidebar.number_input("Component V1", value=0.0)
v2 = st.sidebar.number_input("Component V2", value=0.0)
v3 = st.sidebar.number_input("Component V3", value=0.0)
v4 = st.sidebar.number_input("Component V4", value=0.0)
v5 = st.sidebar.number_input("Component V5", value=0.0)
amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=10.0)

# --- Prediction Logic ---
if st.sidebar.button("🔍 Check Transaction"):
    # Normalize amount just like training data
    scaled_amount = scaler.transform([[amount]])[0][0]
    
    # Combine features into input array
    features = np.array([[v1, v2, v3, v4, v5, scaled_amount]])
    
    # Prediction
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)
    
    # Display Results
    st.subheader("Results")
    if prediction[0] == 1:
        st.error(f"🚨 **Alert:** This transaction is predicted to be **FRAUDULENT**! (Confidence: {prediction_proba[0][1]*100:.2f}%)")
    else:
        st.success(f"✅ **Safe:** This transaction is predicted to be **GENUINE**. (Confidence: {prediction_proba[0][0]*100:.2f}%)")
        
    # Show probabilities
    st.write("---")
    st.write("### Prediction Probability Table")
    prob_df = pd.DataFrame(prediction_proba, columns=["Genuine Probability", "Fraud Probability"])
    st.dataframe(prob_df)
  
