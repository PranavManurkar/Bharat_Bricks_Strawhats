import streamlit as st
import time
import pickle
import pandas as pd
import numpy as np
import os

# --- MODEL CONFIGURATION ---
# Use relative path - works both locally and in deployed app
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "loan_approval_rf.pkl")

# Load model from app folder
@st.cache_resource
def load_model():
    """Load the loan approval model from the app's models folder."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        st.success("✅ Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.error(f"Current working directory: {os.getcwd()}")
        st.error(f"Script location: {os.path.dirname(__file__)}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# --- GROQ LLM CONFIGURATION ---
def initialize_groq():
    """Initialize Groq client for LLM explanations."""
    try:
        import groq
        api_key = "" or st.secrets.get("GROQ_API_KEY", "")
        if api_key:
            return groq.Client(api_key=api_key)
        else:
            st.info("💡 **Tip**: Set GROQ_API_KEY for AI-powered explanations.")
            return None
    except ImportError:
        st.info("💡 groq library not installed. Using rule-based explanations.")
        return None
    except Exception as e:
        st.warning(f"⚠️ Groq error: {str(e)}. Using rule-based explanations.")
        return None

# --- PREDICTION FUNCTIONS ---
def get_loan_prediction(features):
    """Get loan prediction using the loaded model."""
    try:
        model = load_model()
        if model is None:
            return "Error: Model not loaded", 0.0, {}
        
        # Create input dataframe
        input_data = pd.DataFrame([{
            '_education': features['_education'],
            '_self_employed': features['_self_employed'],
            '_no_of_dependents': features['_no_of_dependents'],
            '_income_annum': features['_income_annum'],
            '_loan_amount': features['_loan_amount'],
            '_loan_term': features['_loan_term'],
            '_cibil_score': features['_cibil_score'],
            '_residential_assets_value': features['_residential_assets_value'],
            '_commercial_assets_value': features['_commercial_assets_value'],
            '_luxury_assets_value': features['_luxury_assets_value'],
            '_bank_asset_value': features['_bank_asset_value']
        }])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Get feature importance
        try:
            feature_importance = dict(zip(input_data.columns, model.feature_importances_))
        except:
            feature_importance = {}
        
        # Convert to readable format
        loan_status = "Approved" if prediction == 1 else "Rejected"
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        
        return loan_status, confidence, feature_importance
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return "Error", 0.0, {}

def get_llm_explanation_groq(features, prediction, confidence, feature_importance):
    """Generate AI explanation using Groq LLM."""
    groq_client = initialize_groq()
    
    if groq_client is None:
        return get_rule_based_explanation(features, prediction, confidence)
    
    try:
        # Sort features by importance
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:5]
            feature_text = chr(10).join([f"{i+1}. {feat.replace('_', ' ').title()}: {imp:.3f}" for i, (feat, imp) in enumerate(top_features)])
        else:
            feature_text = "Feature importance not available"
        
        # Compose prompt
        prompt = f"""You are an expert loan approval advisor. Analyze this loan application and provide a clear, actionable explanation.

**Application Details:**
- CIBIL Score: {features['_cibil_score']}
- Annual Income: ₹{features['_income_annum']:,}
- Loan Amount: ₹{features['_loan_amount']:,}
- Loan Term: {features['_loan_term']} months
- Education: {"Graduate" if features['_education'] == 1 else "Not Graduate"}
- Self Employed: {"Yes" if features['_self_employed'] == 1 else "No"}
- Dependents: {features['_no_of_dependents']}
- Residential Assets: ₹{features['_residential_assets_value']:,}
- Commercial Assets: ₹{features['_commercial_assets_value']:,}
- Luxury Assets: ₹{features['_luxury_assets_value']:,}
- Bank Assets: ₹{features['_bank_asset_value']:,}

**Model Prediction:** {prediction}
**Confidence:** {confidence:.1%}

**Top 5 Important Features:**
{feature_text}

**Task:**
1. Explain WHY the model made this prediction
2. If rejected, provide 3 actionable recommendations
3. If approved, mention any risk factors
4. Keep under 200 words, be practical and direct

Professional but friendly tone."""
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.warning(f"⚠️ LLM unavailable: {str(e)}. Using rule-based analysis.")
        return get_rule_based_explanation(features, prediction, confidence)

def get_rule_based_explanation(features, prediction, confidence):
    """Fallback rule-based explanation."""
    cibil = features['_cibil_score']
    income = features['_income_annum']
    loan_amount = features['_loan_amount']
    loan_to_income_ratio = loan_amount / income if income > 0 else 999
    
    if "Approved" in prediction:
        if confidence > 0.9:
            return f"""✅ **Strong Approval** (confidence: {confidence:.1%})

Your application shows excellent indicators:
- CIBIL score of {cibil} is {"excellent" if cibil > 750 else "good"}
- Healthy loan-to-income ratio of {loan_to_income_ratio:.2f}x
- Income of ₹{income:,} provides strong repayment capacity

The model is highly confident in your ability to repay this loan."""
        else:
            return f"""✅ **Conditional Approval** (confidence: {confidence:.1%})

Your application meets basic requirements:
- CIBIL score: {cibil}
- Loan-to-income ratio: {loan_to_income_ratio:.2f}x

Consider providing additional documentation or co-signer."""
    else:
        reasons = []
        recommendations = []
        
        if cibil < 650:
            reasons.append(f"CIBIL score ({cibil}) below threshold")
            recommendations.append("Improve credit score above 650")
        
        if loan_to_income_ratio > 3:
            reasons.append(f"High loan-to-income ratio ({loan_to_income_ratio:.2f}x)")
            recommendations.append("Reduce loan amount or increase income")
        
        if income < 2000000:
            reasons.append("Income below typical threshold")
            recommendations.append("Apply for smaller amount")
        
        reason_text = "\n- ".join(reasons) if reasons else "Multiple risk factors"
        rec_text = "\n".join([f"{i+1}. {r}" for i, r in enumerate(recommendations[:3])])
        default_rec = "1. Improve financial profile\n2. Get co-signer\n3. Apply for smaller amount"
        
        return f"""❌ **Rejection Likely** (confidence: {confidence:.1%})

**Key Issues:**
- {reason_text}

**Recommendations:**
{rec_text if rec_text else default_rec}"""

# --- UI ---

st.set_page_config(page_title="Rural Loan Assistant", layout="wide")

st.title("🌾 Rural Loan Application Assistant")
st.markdown("AI-powered loan approval assessment using ML model and optional LLM explanations.")

with st.expander("ℹ️ Model Information"):
    st.markdown(f"""
    **Model:** Random Forest Classifier  
    **Accuracy:** 98.13%  
    **F1 Score:** 98.12%  
    **Training Data:** 4,269 loan applications  
    **Features:** 11 (CIBIL score, income, assets, etc.)
    
    **LLM:** Groq llama-3.1-8b-instant (optional)
    
    Get free Groq API key: https://console.groq.com/keys
    Set as `GROQ_API_KEY` environment variable for AI explanations.
    """)

tab1, tab2, tab3 = st.tabs(["📋 Personal", "💰 Financial", "🏠 Assets"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        education_encoded = 1 if education == "Graduate" else 0
        self_employed = st.selectbox("Employment", ["Salaried", "Self Employed"])
        self_employed_encoded = 1 if self_employed == "Self Employed" else 0
    with col2:
        no_of_dependents = st.number_input("Dependents", 0, 10, 2)
        cibil_score = st.slider("CIBIL Score", 300, 900, 700, 10)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        annual_income = st.number_input("Annual Income (₹)", 100000, 100000000, 5000000, 100000)
        loan_amount = st.number_input("Loan Amount (₹)", 100000, 100000000, 10000000, 100000)
    with col2:
        loan_term = st.selectbox("Loan Term (months)", [6,12,24,36,60,84,120,180,240,360], index=1)
        if annual_income > 0:
            lti_ratio = loan_amount / annual_income
            st.metric("Loan-to-Income Ratio", f"{lti_ratio:.2f}x")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        residential_assets = st.number_input("Residential Assets (₹)", 0, 100000000, 5000000, 500000)
        commercial_assets = st.number_input("Commercial Assets (₹)", 0, 100000000, 0, 500000)
    with col2:
        luxury_assets = st.number_input("Luxury Assets (₹)", 0, 50000000, 0, 500000)
        bank_assets = st.number_input("Bank Assets (₹)", 0, 50000000, 0, 100000)

st.divider()

application_features = {
    '_education': education_encoded,
    '_self_employed': self_employed_encoded,
    '_no_of_dependents': no_of_dependents,
    '_income_annum': annual_income,
    '_loan_amount': loan_amount,
    '_loan_term': loan_term,
    '_cibil_score': cibil_score,
    '_residential_assets_value': residential_assets,
    '_commercial_assets_value': commercial_assets,
    '_luxury_assets_value': luxury_assets,
    '_bank_asset_value': bank_assets
}

if st.button("🔍 Assess Application", type="primary", use_container_width=True):
    with st.spinner("Loading model and making prediction..."):
        prediction, confidence, feature_importance = get_loan_prediction(application_features)
        time.sleep(0.3)
    
    st.session_state['prediction'] = prediction
    st.session_state['confidence'] = confidence
    st.session_state['feature_importance'] = feature_importance
    st.session_state['features'] = application_features

if 'prediction' in st.session_state:
    st.divider()
    st.subheader("📊 Assessment Results")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if "Approved" in st.session_state['prediction']:
            st.success(f"**✅ Prediction:** {st.session_state['prediction']}")
        else:
            st.error(f"**❌ Prediction:** {st.session_state['prediction']}")
    with col2:
        if st.session_state['confidence'] > 0:
            st.metric("Confidence", f"{st.session_state['confidence']:.1%}")
    
    if st.button("🤖 Get AI Explanation", type="secondary", use_container_width=True):
        with st.spinner("Generating explanation..."):
            explanation = get_llm_explanation_groq(
                st.session_state['features'],
                st.session_state['prediction'],
                st.session_state['confidence'],
                st.session_state['feature_importance']
            )
            st.session_state['explanation'] = explanation
    
    if 'explanation' in st.session_state:
        st.info(f"**🤖 Analysis:**\n\n{st.session_state['explanation']}")
    
    st.markdown("### Key Metrics")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Income", f"₹{application_features['_income_annum']:,.0f}")
    with col_b:
        st.metric("Loan", f"₹{application_features['_loan_amount']:,.0f}")
    with col_c:
        st.metric("CIBIL", f"{application_features['_cibil_score']}")
    with col_d:
        total_assets = (application_features['_residential_assets_value'] + 
                       application_features['_commercial_assets_value'] + 
                       application_features['_luxury_assets_value'] + 
                       application_features['_bank_asset_value'])
        st.metric("Assets", f"₹{total_assets:,.0f}")
    
    st.caption("📊 Random Forest ML Model | 🤖 Groq LLM | Powered by Databricks")
