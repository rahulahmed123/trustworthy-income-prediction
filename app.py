import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trustworthy Income Prediction", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fb;
    }
    h1, h2, h3 {
        color: #1a1a2e;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e6e6e6;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5em 1.5em;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #27ae60;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

model = joblib.load("models/lgbm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
numeric_cols = joblib.load("models/numeric_cols.pkl")

DECISION_THRESHOLD = 0.3494
UNSTABLE_THRESHOLD = 0.5
LGBM_ECE_FROM_PHASE6 = 0.0129
LGBM_BRIER_SCORE = 0.0844
N_PERTURBATION_TRIALS = 20

LOW_UNCERTAINTY_ACCURACY = 0.9886
HIGH_UNCERTAINTY_ACCURACY = 0.7692
GLOBAL_MEAN_STABILITY = 0.9379
GLOBAL_UNSTABLE_SAMPLES = 226

SENSITIVITY_CAPITAL_GAIN = 0.0575
SENSITIVITY_AGE = 0.0028
SENSITIVITY_HOURS = 0.0017

st.title("Trustworthy Income Prediction")
st.caption(
    "Explainability + Reliability + Autonomous Decision-Making demo — "
    "Project 1, Trustworthy Agentic AI"
)

st.header("1. Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 17, 90, 35)
    workclass = st.selectbox(
        "Workclass",
        [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked",
            "Unknown"
        ]
    )
    fnlwgt = st.number_input("Fnlwgt", value=189778)
    education = st.selectbox(
        "Education",
        [
            "Bachelors",
            "Some-college",
            "11th",
            "HS-grad",
            "Prof-school",
            "Assoc-acdm",
            "Assoc-voc",
            "9th",
            "7th-8th",
            "12th",
            "Masters",
            "1st-4th",
            "10th",
            "Doctorate",
            "5th-6th",
            "Preschool"
        ]
    )
    education_num = st.slider("Education Num", 1, 16, 10)

with col2:
    marital_status = st.selectbox(
        "Marital Status",
        [
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse"
        ]
    )
    occupation = st.selectbox(
        "Occupation",
        [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces",
            "Unknown"
        ]
    )
    relationship = st.selectbox(
        "Relationship",
        [
            "Wife",
            "Own-child",
            "Husband",
            "Not-in-family",
            "Other-relative",
            "Unmarried"
        ]
    )
    race = st.selectbox(
        "Race",
        [
            "White",
            "Asian-Pac-Islander",
            "Amer-Indian-Eskimo",
            "Other",
            "Black"
        ]
    )
    sex = st.selectbox("Sex", ["Male", "Female"])

with col3:
    capital_gain = st.number_input("Capital Gain", value=0)
    capital_loss = st.number_input("Capital Loss", value=0)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    native_country = st.selectbox(
        "Native Country",
        ["United-States", "Other"]
    )


def build_raw_input(
    age,
    workclass,
    fnlwgt,
    education,
    education_num,
    marital_status,
    occupation,
    relationship,
    race,
    sex,
    capital_gain,
    capital_loss,
    hours_per_week,
    native_country
):
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

    encoded = pd.get_dummies(
        raw_df,
        columns=categorical_cols,
        drop_first=True
    )

    encoded = encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    encoded[numeric_cols] = scaler.transform(
        encoded[numeric_cols]
    )

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

    raw_input = build_raw_input(
        age,
        workclass,
        fnlwgt,
        education,
        education_num,
        marital_status,
        occupation,
        relationship,
        race,
        sex,
        capital_gain,
        capital_loss,
        hours_per_week,
        native_country
    )

    encoded_input = encode_and_scale(raw_input)

    prediction, probability = predict_with_probability(
        encoded_input
    )

    label = (
        "Income > 50K"
        if prediction == 1
        else "Income <= 50K"
    )

    boundary_distance = abs(
        probability - DECISION_THRESHOLD
    )

    st.header("2. Model Prediction")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Prediction",
        label
    )

    c2.metric(
        "Probability",
        f"{probability:.2%}"
    )

    c3.metric(
        f"Distance from {DECISION_THRESHOLD:.2%} threshold",
        f"{boundary_distance:.2%}"
    )

    st.header("3. Uncertainty (Uncertainty Pillar)")

    uncertainty_score = (
        1 - abs(2 * probability - 1)
    )

    level = uncertainty_level(
        uncertainty_score
    )

    st.metric(
        f"Uncertainty score — Level: {level}",
        f"{uncertainty_score:.3f} / 1.000"
    )

    st.progress(
        min(uncertainty_score, 1.0)
    )

    st.caption(
        "This score is derived directly from how close "
        "the probability is to the decision boundary "
        "(1 = exactly at the boundary, 0 = maximally confident)."
    )

    with st.expander("Model Calibration Reference"):

        col1, col2 = st.columns(2)

        col1.metric(
            "LightGBM ECE",
            f"{LGBM_ECE_FROM_PHASE6:.4f}"
        )

        col2.metric(
            "LightGBM Brier Score",
            f"{LGBM_BRIER_SCORE:.4f}"
        )

        st.caption(
            "These calibration metrics were measured on the "
            "held-out test set and describe the model overall, "
            "not this individual prediction."
        )

    st.header("4. Stability Check (Reliability Pillar)")

    same_count = 0
    prob_changes = []

    rng = np.random.default_rng(42)

    base_encoded = encoded_input.copy()

    for _ in range(N_PERTURBATION_TRIALS):

        trial_encoded = base_encoded.copy()

        trial_encoded["age"] = (
            trial_encoded["age"]
            + rng.normal(0, 0.05)
        )

        trial_encoded["hours_per_week"] = (
            trial_encoded["hours_per_week"]
            + rng.normal(0, 0.05)
        )

        trial_encoded["capital_gain"] = (
            trial_encoded["capital_gain"]
            + rng.normal(0, 0.05)
        )

        trial_prediction, trial_probability = (
            predict_with_probability(trial_encoded)
        )

        if trial_prediction == prediction:
            same_count += 1

        prob_changes.append(
            abs(trial_probability - probability)
        )

    stability_pct = (
        same_count / N_PERTURBATION_TRIALS
    )

    avg_change = np.mean(prob_changes)

    s1, s2, s3 = st.columns(3)

    s1.metric(
        "Stability",
        f"{stability_pct:.0%}",
        f"{same_count}/{N_PERTURBATION_TRIALS} trials unchanged"
    )

    s2.metric(
        "Avg probability change",
        f"{avg_change:.2%}"
    )

    s3.metric(
        "Status",
        "STABLE"
        if stability_pct >= (1 - UNSTABLE_THRESHOLD)
        else "UNSTABLE"
    )

    st.caption(
        "Each trial applies small Gaussian perturbations "
        "to standardized age, hours_per_week, and capital_gain, "
        "following the project's perturbation-based reliability analysis. "
        "A sample is flagged UNSTABLE when its stability score falls "
        "below the same 0.5 cutoff used in Phase 6/7 of the project."
    )

    with st.expander("Project Reliability Reference"):

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Mean Stability",
            f"{GLOBAL_MEAN_STABILITY:.1%}"
        )

        r2.metric(
            "Highly Unstable Samples",
            f"{GLOBAL_UNSTABLE_SAMPLES}"
        )

        r3.metric(
            "Trials per Sample",
            f"{N_PERTURBATION_TRIALS}"
        )

    st.header(
        "5. Explanation (Explainability Pillar — SHAP)"
    )

    explainer = shap.TreeExplainer(model)

    shap_explanation = explainer(
        encoded_input
    )

    shap_row = shap_explanation.values[0]

    if shap_row.ndim > 1:
        shap_row = shap_row[:, 1]

    contrib_df = pd.DataFrame({
        "feature": encoded_input.columns,
        "shap_value": shap_row
    })

    contrib_df["abs_value"] = (
        contrib_df["shap_value"].abs()
    )

    top_contrib = (
        contrib_df
        .sort_values(
            "abs_value",
            ascending=False
        )
        .head(10)
    )

    top_contrib_sorted = (
        top_contrib
        .sort_values("shap_value")
    )

    chart_col, text_col = st.columns([2, 1])

    with chart_col:

        fig, ax = plt.subplots()

        colors = [
            "#ff4b4b" if value < 0
            else "#2ecc71"
            for value in top_contrib_sorted["shap_value"]
        ]

        ax.barh(
            top_contrib_sorted["feature"],
            top_contrib_sorted["shap_value"],
            color=colors
        )

        ax.set_xlabel(
            "SHAP value (impact on prediction)"
        )

        ax.set_title(
            "Top 10 features driving this prediction"
        )

        st.pyplot(fig)

        st.caption(
            "Green bars push the prediction toward a higher "
            "income bracket. Red bars push it toward a lower "
            "income bracket."
        )

    with text_col:

        st.markdown("**Supporting >50K**")

        for _, row in (
            top_contrib[
                top_contrib["shap_value"] > 0
            ]
            .sort_values(
                "shap_value",
                ascending=False
            )
            .iterrows()
        ):
            st.write(
                f"+ {row['feature']}: "
                f"{row['shap_value']:.3f}"
            )

        st.markdown("**Supporting <=50K**")

        for _, row in (
            top_contrib[
                top_contrib["shap_value"] < 0
            ]
            .sort_values("shap_value")
            .iterrows()
        ):
            st.write(
                f"- {row['feature']}: "
                f"{row['shap_value']:.3f}"
            )

    with st.expander("SHAP Waterfall Explanation"):

        shap.plots.waterfall(
            shap_explanation[0],
            max_display=10,
            show=False
        )

        fig_waterfall = plt.gcf()
        fig_waterfall.set_size_inches(7, 4.5)
        fig_waterfall.tight_layout()

        st.pyplot(
            fig_waterfall,
            clear_figure=True
        )

    st.header(
        "6. Decision (Autonomous Decision-Making Pillar)"
    )

    if uncertainty_score >= (1 - UNSTABLE_THRESHOLD):

        decision_mode = "Human Review"
        decision_icon = "⚠️"

    else:

        decision_mode = "Automated Decision"
        decision_icon = "✅"

    d1, d2 = st.columns(2)

    with d1:

        st.markdown("**Model Prediction**")
        st.write(label)
        st.write(
            f"Probability: {probability:.2%}"
        )

    with d2:

        st.markdown("**Final Action**")
        st.write(
            f"{decision_icon} {decision_mode}"
        )

    st.markdown("**Why this decision:**")

    st.write(
        f"- Model probability = {probability:.2%}, "
        f"decision threshold = {DECISION_THRESHOLD:.2%} "
        f"(Phase 4 optimized threshold for LightGBM)"
    )

    st.write(
        f"- Uncertainty score = {uncertainty_score:.3f}, "
        f"review threshold = {(1 - UNSTABLE_THRESHOLD):.3f}"
    )

    st.write(
        f"- Stability across "
        f"{N_PERTURBATION_TRIALS} perturbation trials = "
        f"{stability_pct:.0%}"
    )

    if decision_mode == "Human Review":

        st.write(
            "- Uncertainty meets or exceeds the review "
            "threshold, so this case is not considered "
            "reliable enough for a fully automated decision "
            "and is routed to a human reviewer."
        )

    else:

        st.write(
            "- Uncertainty is below the review threshold, "
            "so this case can be auto-decided under the "
            "current policy."
        )

    st.header("7. Project Validation References")

    v1, v2, v3 = st.columns(3)

    v1.metric(
        "Low-Uncertainty Accuracy",
        f"{LOW_UNCERTAINTY_ACCURACY:.2%}"
    )

    v2.metric(
        "High-Uncertainty Accuracy",
        f"{HIGH_UNCERTAINTY_ACCURACY:.2%}"
    )

    v3.metric(
        "Global Stability",
        f"{GLOBAL_MEAN_STABILITY:.2%}"
    )

    st.caption(
        "These values are project-level validation results "
        "measured on the held-out test set. They are references "
        "for the deployed model and are not recalculated for the "
        "current applicant."
    )

    st.header("8. Perturbation Sensitivity Reference")

    sensitivity_df = pd.DataFrame({
        "Feature": [
            "capital_gain",
            "age",
            "hours_per_week"
        ],
        "Flip Rate": [
            SENSITIVITY_CAPITAL_GAIN,
            SENSITIVITY_AGE,
            SENSITIVITY_HOURS
        ]
    })

    sensitivity_df["Flip Rate"] = (
        sensitivity_df["Flip Rate"]
        .map(lambda x: f"{x:.2%}")
    )

    st.table(sensitivity_df)

    st.caption(
        "Project Phase 7 sensitivity results show that "
        "capital_gain produced substantially more prediction "
        "flips under small perturbations than age or hours_per_week."
    )
