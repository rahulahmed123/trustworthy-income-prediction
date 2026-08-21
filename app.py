import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/lgbm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
numeric_cols = joblib.load("models/numeric_cols.pkl")

st.title("Income Prediction — Trustworthy AI Demo")
st.write("Predicts whether income exceeds $50K/year, using the LightGBM model from Project 1.")

age = st.slider("Age", 17, 90, 35)
workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked", "Unknown"])
fnlwgt = st.number_input("Fnlwgt", value=189778)
education = st.selectbox("Education", ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"])
education_num = st.slider("Education Num", 1, 16, 10)
marital_status = st.selectbox("Marital Status", ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces", "Unknown"])
relationship = st.selectbox("Relationship", ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
race = st.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
sex = st.selectbox("Sex", ["Male", "Female"])
capital_gain = st.number_input("Capital Gain", value=0)
capital_loss = st.number_input("Capital Loss", value=0)
hours_per_week = st.slider("Hours per Week", 1, 99, 40)
native_country = st.selectbox("Native Country", ["United-States", "Other"])

if st.button("Predict"):
    raw_input = pd.DataFrame([{
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education_num": education_num,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
        "native_country": native_country
    }])

    categorical_cols = raw_input.select_dtypes(include="object").columns
    encoded_input = pd.get_dummies(raw_input, columns=categorical_cols)
    encoded_input = encoded_input.reindex(columns=feature_columns, fill_value=0)
    encoded_input[numeric_cols] = scaler.transform(encoded_input[numeric_cols])

    prediction = model.predict(encoded_input)[0]
    probability = model.predict_proba(encoded_input)[0][1]

    if prediction == 1:
        st.success(f"Predicted: Income > $50K (probability: {probability:.2%})")
    else:
        st.info(f"Predicted: Income <= $50K (probability of >$50K: {probability:.2%})")
