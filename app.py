import streamlit as st
import time

# --- MOCK BACKEND FUNCTIONS ---
# These replace your Databricks SDK calls for now

def get_mock_prediction(features):
    # Simulate network delay
    time.sleep(1.5) 
    # Fake logic just for the UI
    if features["income"] > 50000:
        return "Low Risk (Approved)"
    else:
        return "High Risk (Requires Review)"

def get_mock_llm_explanation(features, prediction):
    time.sleep(2)
    if "Approved" in prediction:
        return "The applicant's income and stable crop yield history in this district indicate a strong ability to repay the loan."
    else:
        return "The applicant's requested loan amount is too high relative to their current income. Suggesting a smaller loan or requiring a co-signer would improve approval chances."

# --- UI LAYOUT ---

st.set_page_config(page_title="Rural Loan Assistant", layout="wide")

st.title("🌾 Rural Loan Application Assistant")
st.markdown("Enter the applicant's details. Our system uses regional data (Delta Lake) and AI to assess risk and provide actionable feedback.")

# Use columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Applicant Profile")
    district_id = st.text_input("District ID (For Regional Data lookup)")
    applicant_age = st.number_input("Age", min_value=18, max_value=80, value=35)
    annual_income = st.number_input("Annual Income (INR)", min_value=0, value=45000)
    
with col2:
    st.subheader("Loan Details")
    loan_amount = st.number_input("Requested Loan Amount (INR)", min_value=1000, value=100000)
    primary_crop = st.selectbox("Primary Crop", ["Wheat", "Rice", "Sugarcane", "Cotton", "Other"])

# Submit Button
if st.button("Assess Application", type="primary"):
    
    # Collect all inputs into a dictionary
    application_data = {
        "district": district_id,
        "income": annual_income,
        "loan": loan_amount,
        "crop": primary_crop
    }
    
    st.divider()
    
    # Show loading spinners to make it feel real
    with st.spinner("Querying Decision Tree Model..."):
        risk_status = get_mock_prediction(application_data)
        
    with st.spinner("Generating LLM Explanation in local language..."):
        explanation = get_mock_llm_explanation(application_data, risk_status)
        
    # --- DISPLAY RESULTS ---
    
    st.subheader("Risk Assessment Results")
    
    # Use Streamlit's metric boxes for a clean dashboard look
    if "Approved" in risk_status:
        st.success(f"**Decision Tree Prediction:** {risk_status}")
    else:
        st.error(f"**Decision Tree Prediction:** {risk_status}")
        
    st.info(f"**AI Explanation (Mocked LLM):**\n\n{explanation}")
    
    st.caption("Note: This data is securely logged to the Delta Lake audit table.")