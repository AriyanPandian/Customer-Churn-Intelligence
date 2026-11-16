# Telecom Customer Churn Prediction

**Live demo:** [Open the Streamlit app](https://your-streamlit-community-cloud-url)

## Overview

Telecom providers lose recurring revenue when customers leave without warning.
This project turns customer account and service details into an explainable,
interactive churn-risk assessment, helping retention teams prioritize outreach
and compare potential interventions.

## System architecture

The training pipeline downloads and caches the IBM Telco Customer Churn dataset,
cleans the selected fields, and trains one scikit-learn pipeline containing
preprocessing and an XGBoost classifier. The serialized pipeline and evaluation
metadata are loaded by a Streamlit dashboard. The dashboard produces real-time
risk estimates, counterfactual intervention comparisons, and holdout model
diagnostics.

### Feature engineering

The model uses the same ten-field schema for training and app inference:

- Numeric: `tenure`, `MonthlyCharges`, `TotalCharges`
- Categorical: `Contract`, `InternetService`, `OnlineSecurity`, `TechSupport`,
  `PaymentMethod`, `PaperlessBilling`, `SeniorCitizen`

Missing total charges are estimated from monthly charges and tenure;
`SeniorCitizen` is represented as `"Yes"` / `"No"`. Numeric values are median
imputed and standardized. Categorical values are most-frequent imputed and
one-hot encoded with unknown categories safely ignored.

## Holdout performance

Results below are computed on a stratified 20% test split using random state 42.

| Metric | Score |
| --- | ---: |
| ROC-AUC | 0.8421 |
| Recall | 0.7754 |
| Precision | 0.5235 |
| F1-score | 0.6250 |
| Accuracy | 0.7530 |

## Local quickstart

Requires Python 3.10 or newer.
On macOS, XGBoost also requires the OpenMP runtime (`brew install libomp`).

```bash
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

The dataset is cached at `data/telco_churn.csv`. Training writes
`models/churn_pipeline.pkl` and `models/model_metadata.json`; the app can also
trigger training automatically if either artifact is missing.
