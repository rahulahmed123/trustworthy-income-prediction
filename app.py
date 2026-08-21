import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trustworthy Income Prediction", layout="wide")

model = joblib.load("models/lgbm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
numeric_cols = joblib.load("models/numeric_cols.pkl")

UNCERTAINTY_THRESHOLD = 0.5

st.title("Trustworthy Income Prediction")
st.caption("Explainability + Reliability + Autonomous Decision-Making demo — Project 1, Trustworthy Agentic AI")

st.header("1. Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 17, 90, 35)
    workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked", "Unknown"])
    fnlwgt = st.number_input("Fnlwgt", value=189778)
    education = st.selectbox("Education", ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"])
    education_num = st.slider("Education Num", 1, 16, 10)

with col2:
    marital_status = st.selectbox("Marital Status", ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
    occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces", "Unknown"])
    relationship = st.selectbox("Relationship", ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
    race = st.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
    sex = st.selectbox("Sex", ["Male", "Female"])

with col3:
    capital_gain = st.number_input("Capital Gain", value=0)
    capital_loss = st.number_input("Capital Loss", value=0)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    native_country = st.selectbox("Native Country", ["United-States", "Other"])


def build_raw_input(age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country):
    return pd.DataFrame([{
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


def encode_and_scale(raw_df):
    categorical_cols = raw_df.select_dtypes(include="object").columns
    encoded = pd.get_dummies(raw_df, columns=categorical_cols)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    encoded[numeric_cols] = scaler.transform(encoded[numeric_cols])
    return encoded


def predict_with_probability(encoded_df):
    probability = model.predict_proba(encoded_df)[0][1]
    prediction = int(probability >= 0.5)
    return prediction, probability


if st.button("Run Prediction"):
    raw_input = build_raw_input(age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country)
    encoded_input = encode_and_scale(raw_input)
    prediction, probability = predict_with_probability(encoded_input)

    st.header("2. Prediction")
    if prediction == 1:
        st.success(f"Predicted: Income > $50K  (probability: {probability:.2%})")
    else:
        st.info(f"Predicted: Income <= $50K  (probability of >$50K: {probability:.2%})")

    st.header("3. Uncertainty (Uncertainty Pillar)")
    uncertainty_score = 1 - abs(2 * probability - 1)
    st.metric("Uncertainty score (0 = fully confident, 1 = maximally uncertain)", f"{uncertainty_score:.3f}")
    st.progress(min(uncertainty_score, 1.0))

    if uncertainty_score >= UNCERTAINTY_THRESHOLD:
        st.warning("High uncertainty — this prediction is close to the decision boundary.")
        decision_mode = "Flagged for Human Review"
    else:
        st.success("Low uncertainty — the model is confident in this prediction.")
        decision_mode = "Automated Decision"

    st.header("4. Stability Check (Reliability Pillar)")
    perturbed_raw = raw_input.copy()
    perturbed_raw["capital_gain"] = perturbed_raw["capital_gain"] * 1.05 + 1
    perturbed_raw["hours_per_week"] = perturbed_raw["hours_per_week"] + 1
    perturbed_encoded = encode_and_scale(perturbed_raw)
    perturbed_prediction, perturbed_probability = predict_with_probability(perturbed_encoded)

    if perturbed_prediction != prediction:
        st.warning(f"Prediction FLIPS under a small input change (capital_gain +5%, hours_per_week +1). New probability: {perturbed_probability:.2%}. This case is borderline/unstable.")
    else:
        st.success(f"Prediction is STABLE under a small input change. New probability: {perturbed_probability:.2%} (was {probability:.2%}).")

    st.header("5. Explanation (Explainability Pillar — SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(encoded_input)
    shap_row = shap_values[0]

    contrib_df = pd.DataFrame({
        "feature": encoded_input.columns,
        "shap_value": shap_row
    })
    contrib_df["abs_value"] = contrib_df["shap_value"].abs()
    top_contrib = contrib_df.sort_values("abs_value", ascending=False).head(10)
    top_contrib = top_contrib.sort_values("shap_value")

    fig, ax = plt.subplots()
    colors = ["#ff4b4b" if v < 0 else "#2ecc71" for v in top_contrib["shap_value"]]
    ax.barh(top_contrib["feature"], top_contrib["shap_value"], color=colors)
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title("Top 10 features driving this prediction")
    st.pyplot(fig)
    st.caption("Green = pushes prediction toward >$50K. Red = pushes prediction toward <=$50K.")

    st.header("6. Decision Summary (Autonomous Decision-Making Pillar)")
    st.write(f"**Final decision mode: {decision_mode}**")
    if decision_mode == "Flagged for Human Review":
        st.write("Because uncertainty is high, this case is routed to a human reviewer instead of being auto-approved — matching the human-in-the-loop policy from Phase 6 of the project.")
    else:
        st.write("Because uncertainty is low and the prediction is stable under perturbation, this case can be safely auto-decided without human review.")
