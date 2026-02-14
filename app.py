import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="centered"

)

#----------------- Theme Css -------------------

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}

.section-card {
    padding: 1.2rem;
    border-radius: 12px;
    margin-bottom: 1.2rem;
    border: 1px solid rgba(150,150,150,0.2);
}
</style>
""", unsafe_allow_html=True)

#----------------- Header --------------------

st.markdown("## 🏦 Loan Default Risk Prediction")
st.write("Predict whether customer is Likely Default or not on Loan")

#------------------ Load Model ----------------
import gdown
import os

os.makedirs("models", exist_ok=True)

MODEL_ID = "1_OJhUafKfM5MTdSRVSNvg4G21GthFTxJ"
SCALER_ID = "1I8cNLc2CkMZpVeBWw-4xA83QFL0jnCYk"
ENCODER_ID = "1iKYofX9vfTgX9Mtypn-F9d2l5SFQ7s2i"

if not os.path.exists("models/loan_default_training.pkl"):
    gdown.download(
        f"https://drive.google.com/uc?id={MODEL_ID}",
        "models/loan_default_training.pkl",
        quiet=False
    )

if not os.path.exists("models/scaler.pkl"):
    gdown.download(
        f"https://drive.google.com/uc?id={SCALER_ID}",
        "models/scaler.pkl",
        quiet=False
    )

if not os.path.exists("models/encoders.pkl"):
    gdown.download(
        f"https://drive.google.com/uc?id={ENCODER_ID}",
        "models/encoders.pkl",
        quiet=False
    )
model = pickle.load(open("loan_default_training.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

#------------------ input Section --------------
st.markdown('<div class = "section-card">', unsafe_allow_html=True)
st.subheader("👤 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18,100)
    income = st.number_input("Annual Income", 0)
    credit_score = st.number_input("Credit Score", 300,900)
    months_employed = st.number_input("Months Emoloyed", 0)

with col2:
    loan_amount = st.number_input("Loan Amount", 0)
    interest_rate = st.number_input("Interest Rate %", 0.0)
    dti_ratio = st.number_input("Debt-to-Income Ratio (%)", 0.0) / 100
    loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60])

st.markdown('<div class="section-card">', unsafe_allow_html=True)

#------------------ Financial Details ------------------

st.markdown('<div"class = "section-card">', unsafe_allow_html=True)
st.subheader("💳 Financial Details")

num_credit_lines = st.number_input("Number of Credit Lines", 0) 
has_mortgage = st.selectbox("Has Mortgage?", ["Yes","No"])
has_dependents = st.selectbox("Has Dependents?", ["Yes", "No"]) 
has_cosigner = st.selectbox("Has Co-Signer?", ["Yes", "No"])  

st.markdown('</div>', unsafe_allow_html=True)

#----------------- Convert Catogorical ------------------

has_mortgage = 1 if has_mortgage == "Yes" else 0
has_dependents = 1 if has_dependents == "Yes" else 0 
has_cosigner = 1 if has_cosigner == "Yes" else 0

#-------------------- Personal status----------------

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("📘 Personal Details")

education = st.selectbox("Education", encoders["Education"].classes_)
employment = st.selectbox("Employment Type", encoders["EmploymentType"].classes_)
marital = st.selectbox("Marital Status", encoders["MaritalStatus"].classes_)
loan_purpose = st.selectbox("Loan Purpose", encoders["LoanPurpose"].classes_)

st.markdown('</div>', unsafe_allow_html=True)

# 🔥 NOW ENCODE THEM PROPERLY

education_encoded = encoders["Education"].transform([education])[0]
employment_encoded = encoders["EmploymentType"].transform([employment])[0]
marital_encoded = encoders["MaritalStatus"].transform([marital])[0]
loan_encoded = encoders["LoanPurpose"].transform([loan_purpose])[0]

st.markdown('</div>', unsafe_allow_html=True)
input_data = np.array([[ 
    age,
    income,
    loan_amount,       
    credit_score,       
    months_employed,
    num_credit_lines,
    interest_rate,
    loan_term,
    dti_ratio,
    has_mortgage,
    has_dependents,
    has_cosigner,
    education_encoded,
    employment_encoded,
    marital_encoded,
    loan_encoded
]])

input_data = scaler.transform(input_data)

#------------------ Prediction ------------------------

if st.button("🔍 Predict Loan Default Risk"):

    probability = model.predict_proba(input_data)[0][1] * 100

    st.markdown("### 📊 Default Probability")
    st.write(f"**{probability:.2f}%** ")

    if probability > 50:
        st.error("⚠️ High Risk of Loan Default")
    else:
        st.success("✅ Low Risk of Loan Default")

# -------- Simple Risk Explanation --------
    st.markdown("### 📌 Risk Insights")

    if credit_score < 600:
        st.write("- Low credit score increases risk.")
    if dti_ratio > 0.5:
        st.write("- High debt-to-income ratio indicates financial stress.")
    if months_employed < 12:
        st.write("- Short employment history may increase default risk.")
    if interest_rate > 15:
        st.write("- High interest rate increases repayment burden.")