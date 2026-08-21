# Uncertainty-Aware Explainable Machine Learning for Reliable Decision-Making

**Project 1 of the Trustworthy Agentic AI research roadmap**

## Motivation

Most ML projects pick the "best" model by accuracy or F1 alone. This project asks a different question: is a high-performing model actually *trustworthy*? Trustworthiness is treated here as a multi-dimensional property — explainability, uncertainty, fairness, and reliability all measured separately, because a model can excel on one dimension while quietly failing on another.

## Dataset

UCI Adult Income dataset — binary classification task predicting whether an individual's income exceeds $50K/year based on demographic and employment features.

- Missing values, duplicates, and high-cardinality categories cleaned
- 80/20 stratified train/test split
- `native_country` reduced to US/Other; categoricals one-hot encoded (60 columns); numeric features scaled with `StandardScaler`

## Models

Five models were trained and compared: Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 85.76% | 74.30% | 62.50% | 67.89% | 0.9115 |
| Decision Tree | 86.39% | 78.95% | 59.31% | 67.73% | 0.9039 |
| Random Forest | 86.31% | 80.97% | 56.44% | 66.52% | 0.9175 |
| XGBoost | 87.26% | 76.95% | 67.28% | 71.79% | 0.9304 |
| **LightGBM** | **87.89%** | 79.95% | 66.39% | **72.54%** | **0.9332** |

LightGBM was selected as the primary model for downstream analysis based on top performance and the best/most stable 5-fold cross-validated F1.

## Trustworthiness Evaluation

**1. Fairness** — Optimizing the decision threshold for F1 consistently widened fairness gaps (Sex, Race) across all tree/boosting models, even as overall prediction rates improved. Largest increase observed for XGBoost (+9.09pp).

**2. Explainability (SHAP)** — Feature importance rankings agree strongly within the same model family (LightGBM vs XGBoost, Spearman ρ ≈ 0.96) but less so across families (Random Forest vs boosting, ρ ≈ 0.74–0.76) — explanation reliability is architecture-dependent, not just data-dependent.

**3. Uncertainty** — High-uncertainty samples (~3–15% depending on method) show substantially lower accuracy than low-uncertainty samples (76.9% vs 98.9%), confirming the uncertainty measure tracks real error risk. Uncertainty is also unevenly distributed by group (e.g. higher for male samples than female under LightGBM's margin-based proxy).

**4. Reliability** — Predictions that flip under small input perturbations carry ~3x higher pre-existing uncertainty than stable predictions. `capital_gain` accounts for a disproportionate share of instability relative to its average SHAP importance, suggesting importance and reliability-risk are distinct dimensions.

## Key Finding

> Performance, fairness, explainability, uncertainty, and reliability behave independently and sometimes trade off against each other. Selecting a model on accuracy alone is not sufficient for trustworthy deployment — each dimension needs to be checked on its own.

## Repository Structure

```
.
├── notebooks/      # exploratory + full analysis notebook
├── src/            # reusable scripts (train.py, predict.py, preprocessing.py)
├── models/         # saved trained model + scaler/encoder artifacts
├── data/           # raw/processed data or a link to the source
├── app.py          # Streamlit demo app
├── requirements.txt
└── README.md
```

## Running the Demo Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Roadmap

This is Project 1 of a longer research direction: **Project 1 (this)** → Project 2 (Robust XAI) → Deep Learning + LLM Fundamentals bridge → Project 3 (Agent + XAI + Uncertainty) → **Final: Trustworthy Agentic AI**

## Limitations

- Uncertainty measures were architecture-specific (ensemble disagreement for Random Forest vs. probability-margin proxy for LightGBM) and are not directly comparable across model families
- Reliability testing used random perturbations only — adversarial robustness was not tested
- Only SHAP was used for explainability; cross-method (e.g. LIME) consistency was not evaluated
- Analysis is based on a single train/test split, aside from 5-fold CV in the evaluation phase
