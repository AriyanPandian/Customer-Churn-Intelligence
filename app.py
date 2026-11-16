"""Streamlit dashboard for interactive telecom customer churn predictions."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_pipeline.pkl"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
TRAIN_SCRIPT = ROOT / "src" / "train.py"

st.set_page_config(page_title="Customer Churn Intelligence", layout="wide")


def ensure_assets() -> None:
    """Run the training pipeline if either deployment asset is missing."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        with st.spinner("Training the churn model for first-time setup..."):
            subprocess.run(
                [sys.executable, str(TRAIN_SCRIPT)],
                cwd=ROOT,
                check=True,
            )


@st.cache_resource
def load_pipeline() -> Any:
    import joblib

    ensure_assets()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata() -> dict[str, Any]:
    ensure_assets()
    with METADATA_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def make_input(
    tenure: int,
    monthly_charges: float,
    total_charges: float,
    contract: str,
    internet_service: str,
    online_security: str,
    tech_support: str,
    payment_method: str,
    paperless_billing: str,
    senior_citizen: str,
) -> pd.DataFrame:
    """Create one inference row in the same schema and order used for training."""
    return pd.DataFrame(
        [
            {
                "tenure": tenure,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "Contract": contract,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "TechSupport": tech_support,
                "PaymentMethod": payment_method,
                "PaperlessBilling": paperless_billing,
                "SeniorCitizen": senior_citizen,
            }
        ]
    )


pipeline = load_pipeline()
metadata = load_metadata()
baseline = float(metadata["baseline_churn_rate"])

st.title("Customer Churn Intelligence")
st.caption("Assess customer risk, explore retention actions, and review model performance.")

with st.sidebar:
    st.header("Customer profile")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.slider(
        "Monthly charges ($)",
        min_value=18.0,
        max_value=120.0,
        value=70.0,
        step=0.5,
    )
    total_charges = st.number_input(
        "Total charges ($)",
        min_value=0.0,
        value=round(tenure * monthly_charges, 2),
        step=1.0,
    )
    internet_service = st.selectbox(
        "Internet service", ["Fiber optic", "DSL", "No"]
    )
    online_security = st.selectbox(
        "Online security", ["No", "Yes", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech support", ["No", "Yes", "No internet service"]
    )
    payment_method = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
    senior_citizen = st.selectbox("Senior citizen", ["No", "Yes"])

customer = make_input(
    tenure,
    monthly_charges,
    total_charges,
    contract,
    internet_service,
    online_security,
    tech_support,
    payment_method,
    paperless_billing,
    senior_citizen,
)
probability = float(pipeline.predict_proba(customer)[0, 1])
risk_tier = "Low" if probability < 0.35 else "Medium" if probability <= 0.65 else "High"

risk_tab, performance_tab = st.tabs(
    ["Real-Time Risk & Retention Engine", "Model Performance & Feature Drivers"]
)

with risk_tab:
    st.subheader("Customer churn assessment")
    kpi_probability, kpi_tier, kpi_revenue = st.columns(3)
    kpi_probability.metric(
        "Churn probability",
        f"{probability:.1%}",
        delta=f"{probability - baseline:+.1%} vs. baseline",
        delta_color="inverse",
    )
    kpi_tier.metric("Risk tier", risk_tier)
    kpi_revenue.metric(
        "Annual revenue at risk", f"${monthly_charges * 12:,.2f}"
    )

    gauge_column, explanation_column = st.columns([1, 1])
    with gauge_column:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%"},
                title={"text": "Predicted churn risk"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#243b53"},
                    "steps": [
                        {"range": [0, 35], "color": "#b7e4c7"},
                        {"range": [35, 65], "color": "#ffe08a"},
                        {"range": [65, 100], "color": "#f4a6a6"},
                    ],
                    "threshold": {
                        "line": {"color": "#243b53", "width": 4},
                        "thickness": 0.75,
                        "value": probability * 100,
                    },
                },
            )
        )
        gauge.update_layout(height=310, margin={"t": 55, "b": 10, "l": 25, "r": 25})
        st.plotly_chart(gauge, use_container_width=True)
    with explanation_column:
        st.markdown("#### Risk interpretation")
        st.write(
            f"This profile is **{risk_tier.lower()} risk**. Its predicted churn "
            f"probability is {probability:.1%}, compared with the historical "
            f"dataset baseline of {baseline:.1%}."
        )
        st.write(
            f"The annualized recurring revenue represented by this account is "
            f"**${monthly_charges * 12:,.2f}**."
        )

    st.markdown("#### What-if retention simulator")
    interventions = [
        ("Switch to a one-year contract", {"Contract": "One year"}),
        ("Add tech support", {"TechSupport": "Yes"}),
        ("Add online security", {"OnlineSecurity": "Yes"}),
        (
            "Switch to automatic credit card",
            {"PaymentMethod": "Credit card (automatic)"},
        ),
    ]
    scenario_results: list[dict[str, Any]] = []
    for label, changes in interventions:
        scenario = customer.copy()
        for column, value in changes.items():
            scenario.loc[0, column] = value
        scenario_probability = float(pipeline.predict_proba(scenario)[0, 1])
        scenario_results.append(
            {
                "intervention": label,
                "probability": scenario_probability,
                "reduction": probability - scenario_probability,
            }
        )

    simulator_column, recommendations_column = st.columns([3, 2])
    with simulator_column:
        chart = go.Figure(
            go.Bar(
                x=[item["reduction"] * 100 for item in scenario_results],
                y=[item["intervention"] for item in scenario_results],
                orientation="h",
                marker_color=[
                    "#2a9d8f" if item["reduction"] > 0 else "#e76f51"
                    for item in scenario_results
                ],
                text=[f"{item['reduction']:+.1%}" for item in scenario_results],
                textposition="outside",
                customdata=[item["probability"] for item in scenario_results],
                hovertemplate=(
                    "%{y}<br>Churn probability change: %{x:+.1f} pp"
                    "<br>Scenario probability: %{customdata:.1%}<extra></extra>"
                ),
            )
        )
        chart.add_vline(x=0, line_color="#52606d", line_width=1)
        chart.update_layout(
            xaxis_title="Reduction in churn probability (percentage points)",
            yaxis_title=None,
            height=330,
            margin={"l": 10, "r": 35, "t": 20, "b": 40},
        )
        st.plotly_chart(chart, use_container_width=True)

    with recommendations_column:
        st.markdown("##### Recommended actions")
        positive_scenarios = sorted(
            (item for item in scenario_results if item["reduction"] > 0),
            key=lambda item: item["reduction"],
            reverse=True,
        )
        if positive_scenarios:
            for item in positive_scenarios[:3]:
                st.write(
                    f"- **{item['intervention']}** — modeled reduction "
                    f"of {item['reduction']:.1%}."
                )
        else:
            st.write(
                "The tested changes did not lower this profile's modeled risk. "
                "Consider a personalized service review and monitor engagement "
                "before applying an incentive."
            )
        st.caption("Scenario estimates are directional model outputs, not guarantees.")

with performance_tab:
    metrics = metadata["metrics"]
    st.subheader("Holdout test set")
    metric_columns = st.columns(4)
    for column, label, key in zip(
        metric_columns,
        ["ROC-AUC", "Recall", "Precision", "F1-Score"],
        ["roc_auc", "recall", "precision", "f1"],
        strict=True,
    ):
        column.metric(label, f"{metrics[key]:.3f}")

    chart_column, matrix_column = st.columns(2)
    with chart_column:
        importance = metadata["top_feature_importances"][:10]
        feature_chart = go.Figure(
            go.Bar(
                x=[item["importance"] for item in reversed(importance)],
                y=[item["feature"] for item in reversed(importance)],
                orientation="h",
                marker_color="#3282b8",
            )
        )
        feature_chart.update_layout(
            title="Top churn drivers",
            xaxis_title="Aggregated feature importance",
            yaxis_title=None,
            height=430,
            margin={"l": 15, "r": 15, "t": 55, "b": 40},
        )
        st.plotly_chart(feature_chart, use_container_width=True)
    with matrix_column:
        matrix = metadata["confusion_matrix"]
        z_values = [[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]]
        heatmap = go.Figure(
            go.Heatmap(
                z=z_values,
                x=["Predicted: No churn", "Predicted: Churn"],
                y=["Actual: No churn", "Actual: Churn"],
                colorscale="Blues",
                showscale=True,
                text=z_values,
                texttemplate="%{text}",
                hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>",
            )
        )
        heatmap.update_layout(
            title="Test set confusion matrix",
            height=430,
            margin={"l": 15, "r": 15, "t": 55, "b": 60},
        )
        st.plotly_chart(heatmap, use_container_width=True)
    st.caption(
        f"Evaluation uses a stratified 20% holdout from "
        f"{metadata['dataset_rows']:,} customer records."
    )
