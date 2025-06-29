import streamlit as st
import joblib
import sqlite3
import pandas as pd

# Load model
model = joblib.load('model/churn_model.pkl')

# Connect to DB
conn = sqlite3.connect('data/customers.db')
cursor = conn.cursor()

st.title("📉 Customer Churn Predictor")

# Input form
with st.form("predict_form"):
    customer_id = st.text_input("Customer ID")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    monthly = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
    total = st.number_input("Total Charges", min_value=0.0, step=1.0)
    
    submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare input
        gender_val = 1 if gender == "Male" else 0
        senior_val = 1 if senior == "Yes" else 0
        input_data = [[gender_val, senior_val, tenure, monthly, total]]

        # Predict
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction:")
        if prediction == 1:
            st.error(f"⚠️ Likely to Churn (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Not Likely to Churn (Probability: {prob:.2f})")

        # Store in DB
        cursor.execute("""
            INSERT INTO customers (customerID, gender, SeniorCitizen, tenure, MonthlyCharges, TotalCharges, Churn)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (customer_id, gender, senior_val, tenure, monthly, total, int(prediction)))
        conn.commit()

# Show table
st.markdown("### 📊 Stored Customers")
df = pd.read_sql("SELECT * FROM customers ORDER BY ROWID DESC LIMIT 10", conn)
st.dataframe(df)
