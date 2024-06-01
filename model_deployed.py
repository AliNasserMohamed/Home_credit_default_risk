import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the XGBoost model
model = pickle.load(open("xgboost_model.pkl", "rb"))


# Function to predict the risk category
def predict_new_user(model, feature_values):
    feature_values["RATE_OF_LOAN"] = feature_values["AMT_ANNUITY"] / feature_values["AMT_CREDIT"]

    feature_values["AMT_INCOME_TOTAL"] = np.log(feature_values["AMT_INCOME_TOTAL"])
    feature_values["AMT_CREDIT"] = np.log(feature_values["AMT_CREDIT"])
    feature_values["AMT_ANNUITY"] = np.log(feature_values["AMT_ANNUITY"])
    feature_values["AMT_GOODS_PRICE"] = np.log(feature_values["AMT_GOODS_PRICE"])

    # Ensure the feature values are provided in the correct order
    expected_features = ['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', "NAME_INCOME_TYPE",
                         "REG_CITY_NOT_WORK_CITY", 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                         'AMT_GOODS_PRICE', 'AGE_YEARS', 'YEARS_EMPLOYED', 'YEARS_REGISTRATION',
                         'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', "RATE_OF_LOAN"]

    # Create a DataFrame for the single instance
    input_df = pd.DataFrame([feature_values], columns=expected_features)

    # Predict using the model
    pred_prob = model.predict_proba(input_df)[0]

    # Determine risk category based on the predicted probability
    risk_category = "Low Risk" if pred_prob[-1] < 0.2 else "High Risk"
    return risk_category, pred_prob[-1]


st.header("Loan Repayment Prediction Dashboard")

# Input fields in two columns
col1, col2 = st.columns(2)

with col1:
    code_gender = st.radio("Gender", ["M", "F"], index=0)

    flag_own_realty = st.radio("Owns Realty", ["Yes", "No"], index=0)

    name_income_type = st.selectbox("Income Type", ["Working", "State servant", "Commercial associate",
                                                    "Pensioner", "Unemployed", "Student", "Businessman",
                                                    "Maternity leave"], index=0)
    ext_source_1 = st.number_input("External Source 1", value=0.0)
    ext_source_2 = st.number_input("External Source 2", value=0.0)
    ext_source_3 = st.number_input("External Source 3", value=0.0)
    years_employed = st.number_input("Years Employed", min_value=0, value=0)
    years_registration = st.number_input("Years Registration", min_value=0, value=0)

with col2:
    flag_own_car = st.radio("Owns a Car", ["Yes", "No"], index=0)
    reg_city_not_work_city = st.radio("Reg City Not Work City", ["Yes", "No"], index=0)
    name_education_type = st.selectbox("Education Type", ["Higher education", "Secondary / secondary special",
                                                          "Incomplete higher", "Lower secondary", "Academic degree"],
                                       index=0)

    cnt_children = st.number_input("Number of Children", min_value=0, value=0)
    amt_income_total = st.number_input("Total Income", min_value=0, value=0)
    amt_credit = st.number_input("Credit Amount", min_value=0, value=0)
    amt_annuity = st.number_input("Annuity Amount", min_value=0, value=0)
    amt_goods_price = st.number_input("Goods Price", min_value=0, value=0)
age_years = st.number_input("Age (years)", min_value=0, value=0)

if st.button("Predict"):
    feature_values = {
        'CODE_GENDER': code_gender,
        'NAME_EDUCATION_TYPE': name_education_type,
        'FLAG_OWN_CAR': 1 if flag_own_car == "Yes" else 0,
        'FLAG_OWN_REALTY': 1 if flag_own_realty == "Yes" else 0,
        'NAME_INCOME_TYPE': name_income_type,
        'REG_CITY_NOT_WORK_CITY': 1 if reg_city_not_work_city == "Yes" else 0,
        'CNT_CHILDREN': cnt_children,
        'AMT_INCOME_TOTAL': amt_income_total,
        'AMT_CREDIT': amt_credit,
        'AMT_ANNUITY': amt_annuity,
        'AMT_GOODS_PRICE': amt_goods_price,
        'AGE_YEARS': age_years,
        'YEARS_EMPLOYED': years_employed,
        'YEARS_REGISTRATION': years_registration,
        'EXT_SOURCE_1': ext_source_1,
        'EXT_SOURCE_2': ext_source_2,
        'EXT_SOURCE_3': ext_source_3
    }

    risk_category, proba = predict_new_user(model, feature_values)
    # Assuming `risk_category` is the variable holding the risk category value
    if risk_category == "High Risk":
        st.markdown(
            f'<p style="color:red;"> {risk_category} with probability of not repaying of {proba}</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<p>{risk_category} with probability of not repaying of {proba}</p>',
            unsafe_allow_html=True
        )
