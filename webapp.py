import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

# Load the saved model and preprocessing pipeline
pipeline = joblib.load('credit_risk_pipeline.joblib')
model = pipeline.named_steps['classifier']
preprocessor = pipeline.named_steps['preprocessor']

# Page configuration
st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

# Title
st.title("📊 Credit Risk Prediction System")

# ---- Input Section ----
st.markdown("## 📝 Applicant Information")
with st.form(key='credit_risk_form'):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=18, max_value=100, value=25)
        sex = st.selectbox('Sex', ['male', 'female'])
        housing = st.selectbox('Housing', ['own', 'rent', 'free'])

    with col2:
        job = st.selectbox('Job (0 = Unskilled, 3 = Highly Skilled)', [0, 1, 2, 3])
        saving_accounts = st.selectbox('Saving accounts', ['little', 'moderate', 'quite rich', 'rich', 'missing'])
        credit_amount = st.number_input('Credit amount (Loan requested)', min_value=0, value=10000)

    with col3:
        checking_account = st.selectbox('Checking account', ['little', 'moderate', 'rich', 'none', 'missing'])
        duration = st.number_input('Duration (Loan duration in months)', min_value=1, value=24)
        purpose = st.selectbox('Purpose', ['radio/TV', 'education', 'car', 'furniture/equipment'])

    submit_button = st.form_submit_button(label='🔍 Submit & Predict')

# ---- Prediction & Results ----
if submit_button:
    st.markdown("---")
    st.markdown("## 🧾 Prediction Results")

    input_data = {
        'Age': age,
        'Sex': sex,
        'Job': job,
        'Housing': housing,
        'Saving accounts': saving_accounts,
        'Checking account': checking_account,
        'Credit amount': credit_amount,
        'Duration': duration,
        'Purpose': purpose
    }

    # Make prediction
    input_df = pd.DataFrame([input_data])
    transformed_input = preprocessor.transform(input_df)
    prediction = model.predict(transformed_input)
    result = "Good Credit Risk" if prediction[0] == 1 else "Bad Credit Risk"

    # Display result with color
    if result == "Good Credit Risk":
        st.success(f"🎯 **Prediction:** {result}", icon="✅")
    else:
        st.error(f"🎯 **Prediction:** {result}", icon="❌")

    # ---- SHAP Explanation ----
    st.markdown("### 🔍 SHAP Explanation (Model Reasoning)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed_input)

    if isinstance(shap_values, list):
        shap_explanation = shap_values[1][0]  # For Class 1 → 'Good Credit Risk'
    else:
        shap_explanation = shap_values[0]

    feature_names = preprocessor.get_feature_names_out()
    green_factors = []
    red_factors = []

    for feature, value in zip(feature_names, shap_explanation):
        if value > 0:
            green_factors.append(f"{feature} (Positive Impact)")
        elif value < 0:
            red_factors.append(f"{feature} (Negative Impact)")

    # Show only the top 3 most influential factors (if any)
    st.markdown("#### 🟢 Green Factors (Positive Impact)")
    if green_factors:
        for factor in green_factors[:3]:
            st.markdown(f"<div style='color:green'>{factor}</div>", unsafe_allow_html=True)
    else:
        st.write("None")

    st.markdown("#### 🔴 Red Factors (Negative Impact)")
    if red_factors:
        for factor in red_factors[:3]:
            st.markdown(f"<div style='color:red'>{factor}</div>", unsafe_allow_html=True)
    else:
        st.write("None")

    # ---- Feature Importance ----
    st.markdown("### 📌 Feature Importance")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(importances)), importances[indices], align="center")
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Relative Importance")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    # ---- Report Download ----
    st.markdown("### 📥 Download Report")
    prediction_report = pd.DataFrame([input_data])
    prediction_report['Prediction'] = result
    csv = prediction_report.to_csv(index=False)
    st.download_button(
        label="Download Prediction Report as CSV",
        data=csv,
        file_name='prediction_report.csv',
        mime='text/csv'
    )
