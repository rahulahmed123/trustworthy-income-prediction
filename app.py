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

DECISION_THRESHOLD = 0.5
UNCERTAINTY_THRESHOLD = 0.5
LGBM_ECE_FROM_PHASE6 = 0.0129
N_PERTURBATION_TRIALS = 20

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
    prediction = int(probability >= DECISION_THRESHOLD)
    return prediction, probability


def uncertainty_level(score):
    if score < 0.3:
        return "LOW"
    elif score < 0.5:
        return "MEDIUM"
    else:
        return "HIGH"


if st.button("Run Prediction"):
    raw_input = build_raw_input(age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country)
    encoded_input = encode_and_scale(raw_input)
    prediction, probability = predict_with_probability(encoded_input)

    st.header("2. Model Prediction")
    label = "Income > $50K" if prediction == 1 else "Income <= $50K"
    boundary_distance = abs(probability - DECISION_THRESHOLD)
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", label)
    c2.metric("Probability", f"{probability:.2%}")
    c3.metric("Distance from 50% boundary", f"{boundary_distance:.2%}")

    st.header("3. Uncertainty (Uncertainty Pillar)")
    uncertainty_score = 1 - abs(2 * probability - 1)
    level = uncertainty_level(uncertainty_score)
    st.metric(f"Uncertainty score — Level: {level}", f"{uncertainty_score:.3f} / 1.000")
    st.progress(min(uncertainty_score, 1.0))
    st.caption(
        "This score is derived directly from how close the probability is to the 50% "
        "decision boundary (1 = exactly at the boundary, 0 = maximally confident)."
    )
    st.info(
        f"Model calibration reference (measured on the held-out test set in Phase 6): "
        f"LightGBM Expected Calibration Error = {LGBM_ECE_FROM_PHASE6}. "
        f"This describes the model's calibration in general, not a guarantee about this single prediction."
    )

    st.header("4. Stability Check (Reliability Pillar)")
    same_count = 0
    prob_changes = []
    rng = np.random.default_rng(42)
    for _ in range(N_PERTURBATION_TRIALS):
        trial_raw = raw_input.copy()
        gain_noise = rng.normal(loc=0, scale=max(abs(capital_gain) * 0.05, 50))
        hours_noise = rng.integers(-2, 3)
        trial_raw["capital_gain"] = max(trial_raw["capital_gain"].iloc[0] + gain_noise, 0)
        trial_raw["hours_per_week"] = min(max(trial_raw["hours_per_week"].iloc[0] + hours_noise, 1), 99)
        trial_encoded = encode_and_scale(trial_raw)
        trial_prediction, trial_probability = predict_with_probability(trial_encoded)
        if trial_prediction == prediction:
            same_count += 1
        prob_changes.append(abs(trial_probability - probability))

    stability_pct = same_count / N_PERTURBATION_TRIALS
    avg_change = np.mean(prob_changes)

    s1, s2, s3 = st.columns(3)
    s1.metric("Stability", f"{stability_pct:.0%}", f"{same_count}/{N_PERTURBATION_TRIALS} trials unchanged")
    s2.metric("Avg probability change", f"{avg_change:.2%}")
    s3.metric("Status", "STABLE" if stability_pct >= 0.8 else "UNSTABLE")
    st.caption(f"Each trial randomly perturbs capital_gain and hours_per_week by a small amount, based on the Phase 7 finding that capital_gain drives most prediction instability.")

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
    top_contrib_sorted = top_contrib.sort_values("shap_value")

    chart_col, text_col = st.columns([2, 1])

    with chart_col:
        fig, ax = plt.subplots()
        colors = ["#ff4b4b" if v < 0 else "#2ecc71" for v in top_contrib_sorted["shap_value"]]
        ax.barh(top_contrib_sorted["feature"], top_contrib_sorted["shap_value"], color=colors)
        ax.set_xlabel("SHAP value (impact on prediction)")
        ax.set_title("Top 10 features driving this prediction")
        st.pyplot(fig)
        st.caption("Green bars push the prediction toward a higher income bracket. Red bars push it toward a lower income bracket.")

    with text_col:
        st.markdown("**Supporting >50K**")
        for _, row in top_contrib[top_contrib["shap_value"] > 0].sort_values("shap_value", ascending=False).iterrows():
            st.write(f"+ {row['feature']}: {row['shap_value']:.3f}")
        st.markdown("**Supporting <=50K**")
        for _, row in top_contrib[top_contrib["shap_value"] < 0].sort_values("shap_value").iterrows():
            st.write(f"- {row['feature']}: {row['shap_value']:.3f}")

    st.header("6. Decision (Autonomous Decision-Making Pillar)")
    if uncertainty_score >= UNCERTAINTY_THRESHOLD:
        decision_mode = "Human Review"
        decision_icon = "⚠️"
    else:
        decision_mode = "Automated Decision"
        decision_icon = "✅"

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Model Prediction**")
        st.write(label)
        st.write(f"Probability: {probability:.2%}")
    with d2:
        st.markdown("**Final Action**")
        st.write(f"{decision_icon} {decision_mode}")

    st.markdown("**Why this decision:**")
    st.write(f"- Model probability = {probability:.2%}, decision threshold = {DECISION_THRESHOLD:.0%}")
    st.write(f"- Uncertainty score = {uncertainty_score:.3f}, autonomous-use threshold = {UNCERTAINTY_THRESHOLD:.3f}")
    st.write(f"- Stability across {N_PERTURBATION_TRIALS} perturbation trials = {stability_pct:.0%}")
    if decision_mode == "Human Review":
        st.write("- Uncertainty meets or exceeds the autonomous-use threshold, so this case is not considered reliable enough for a fully automated decision and is routed to a human reviewer.")
    else:
        st.write("- Uncertainty is below the autonomous-use threshold, so this case can be auto-decided under the current policy.")
