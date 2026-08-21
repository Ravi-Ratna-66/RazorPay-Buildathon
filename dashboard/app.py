import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# Add project root to Python path
PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# Import RecoverX components
from ml.predict import predict_recovery

from agents.diagnosis_agent import (
    diagnose_transaction
)

from agents.decision_agent import (
    decide_recovery_action
)

from agents.guardrails import (
    validate_action
)

from agents.action_executor import (
    execute_action
)


# File paths
SIMULATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "recovery_simulation.csv"
)

BASELINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "baseline_comparison.csv"
)


# Page configuration
st.set_page_config(
    page_title="RecoverX",
    page_icon="💳",
    layout="wide"
)


# Load data
@st.cache_data
def load_simulation_data():

    return pd.read_csv(
        SIMULATION_PATH
    )


@st.cache_data
def load_baseline_data():

    return pd.read_csv(
        BASELINE_PATH
    )


df = load_simulation_data()

baseline_df = load_baseline_data()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("RecoverX")

st.subheader(
    "Autonomous Revenue Recovery"
)

st.write(
    "AI-powered recovery intelligence for "
    "failed payments and abandoned checkouts."
)

st.divider()


# --------------------------------------------------
# Overview Metrics
# --------------------------------------------------

transactions = len(df)

revenue_at_risk = float(
    df["amount"].sum()
)

recovered_revenue = float(
    df[
        "simulated_recovered_revenue"
    ].sum()
)

recovered_transactions = int(
    df[
        "simulated_recovered"
    ].sum()
)

if revenue_at_risk > 0:

    revenue_recovery_rate = (
        recovered_revenue
        / revenue_at_risk
        * 100
    )

else:

    revenue_recovery_rate = 0.0


if transactions > 0:

    transaction_recovery_rate = (
        recovered_transactions
        / transactions
        * 100
    )

else:

    transaction_recovery_rate = 0.0


# KPI cards

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Revenue at Risk",
        f"₹{revenue_at_risk / 10000000:.2f} Cr"
    )


with col2:

    st.metric(
        "Recovered Revenue",
        f"₹{recovered_revenue / 10000000:.2f} Cr"
    )


with col3:

    st.metric(
        "Revenue Recovery",
        f"{revenue_recovery_rate:.2f}%"
    )


with col4:

    st.metric(
        "Transaction Recovery",
        f"{transaction_recovery_rate:.2f}%"
    )


# --------------------------------------------------
# Recovery Analysis
# --------------------------------------------------

st.divider()

st.header(
    "Recovery Analysis"
)


# Failure reason data

failure_df = (
    df.groupby(
        "failure_reason"
    )
    .agg(
        revenue_at_risk=(
            "amount",
            "sum"
        ),
        recovered_revenue=(
            "simulated_recovered_revenue",
            "sum"
        )
    )
    .reset_index()
)


failure_df["recovery_rate"] = (
    failure_df[
        "recovered_revenue"
    ]
    /
    failure_df[
        "revenue_at_risk"
    ]
    * 100
)


# Payment method data

payment_df = (
    df.groupby(
        "payment_method"
    )
    .agg(
        revenue_at_risk=(
            "amount",
            "sum"
        ),
        recovered_revenue=(
            "simulated_recovered_revenue",
            "sum"
        )
    )
    .reset_index()
)


payment_df["recovery_rate"] = (
    payment_df[
        "recovered_revenue"
    ]
    /
    payment_df[
        "revenue_at_risk"
    ]
    * 100
)


# Two charts side by side

col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "Recovery by Failure Reason"
    )

    fig_failure = px.bar(
        failure_df,
        x="failure_reason",
        y="recovery_rate",
        title="Recovery Rate (%)",
        labels={
            "failure_reason":
                "Failure Reason",

            "recovery_rate":
                "Recovery Rate (%)"
        }
    )

    fig_failure.update_layout(
        xaxis_tickangle=-35
    )

    st.plotly_chart(
        fig_failure,
        use_container_width=True
    )


with col2:

    st.subheader(
        "Recovery by Payment Method"
    )

    fig_payment = px.bar(
        payment_df,
        x="payment_method",
        y="recovery_rate",
        title="Recovery Rate (%)",
        labels={
            "payment_method":
                "Payment Method",

            "recovery_rate":
                "Recovery Rate (%)"
        }
    )

    st.plotly_chart(
        fig_payment,
        use_container_width=True
    )


# --------------------------------------------------
# Recovery Actions
# --------------------------------------------------

st.subheader(
    "Recovery Actions"
)


action_df = (
    df[
        "recommended_action"
    ]
    .value_counts()
    .reset_index()
)


action_df.columns = [
    "action",
    "transactions"
]


fig_actions = px.bar(
    action_df,
    x="action",
    y="transactions",
    title="Recovery Actions Recommended",
    labels={
        "action":
            "Action",

        "transactions":
            "Number of Transactions"
    }
)


fig_actions.update_layout(
    xaxis_tickangle=-30
)


st.plotly_chart(
    fig_actions,
    use_container_width=True
)


# --------------------------------------------------
# Baseline vs RecoverX
# --------------------------------------------------

st.divider()

st.header(
    "Baseline vs RecoverX"
)


baseline_revenue = float(
    baseline_df[
        "baseline_recovered_revenue"
    ].sum()
)

naive_revenue = float(
    baseline_df[
        "naive_recovered_revenue"
    ].sum()
)

recoverx_revenue = float(
    baseline_df[
        "simulated_recovered_revenue"
    ].sum()
)


revenue_uplift = (
    recoverx_revenue
    - naive_revenue
)


if naive_revenue > 0:

    uplift_percentage = (
        revenue_uplift
        / naive_revenue
        * 100
    )

else:

    uplift_percentage = 0.0


comparison_df = pd.DataFrame(
    {
        "Strategy": [
            "Baseline",
            "Naive Recovery",
            "RecoverX"
        ],

        "Recovered Revenue": [
            baseline_revenue,
            naive_revenue,
            recoverx_revenue
        ]
    }
)


fig_comparison = px.bar(
    comparison_df,
    x="Strategy",
    y="Recovered Revenue",
    title="Recovered Revenue Comparison",
    labels={
        "Strategy":
            "Strategy",

        "Recovered Revenue":
            "Recovered Revenue (₹)"
    },

    text_auto=".2s"
)


st.plotly_chart(
    fig_comparison,
    use_container_width=True
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Naive Recovery",
        f"₹{naive_revenue / 100000:.2f} L"
    )


with col2:

    st.metric(
        "RecoverX Recovery",
        f"₹{recoverx_revenue / 100000:.2f} L"
    )


with col3:

    st.metric(
        "Revenue Uplift",
        f"₹{revenue_uplift / 100000:.2f} L",
        f"{uplift_percentage:.2f}%"
    )


st.caption(
    "Comparison is based on the synthetic simulation dataset "
    "and should not be interpreted as production performance."
)


# --------------------------------------------------
# Live RecoverX Agent
# --------------------------------------------------

st.divider()

st.header(
    "Live RecoverX Agent"
)

st.write(
    "Select a failed transaction and run the "
    "complete RecoverX recovery workflow."
)


# Only failed and abandoned transactions

failed_transactions = df[
    df[
        "payment_status"
    ].isin(
        [
            "failed",
            "abandoned"
        ]
    )
].copy()


# Transaction selection

transaction_options = (
    failed_transactions[
        "transaction_id"
    ]
    .astype(str)
    .tolist()
)


selected_transaction_id = st.selectbox(
    "Select a transaction",
    transaction_options
)


# Find selected transaction

selected_row = (
    failed_transactions[
        failed_transactions[
            "transaction_id"
        ].astype(str)
        ==
        selected_transaction_id
    ]
    .iloc[0]
)


# --------------------------------------------------
# Selected Transaction Information
# --------------------------------------------------

st.subheader(
    "Transaction Details"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Transaction ID",
        str(
            selected_row[
                "transaction_id"
            ]
        )
    )


with col2:

    st.metric(
        "Amount",
        f"₹{float(selected_row['amount']):,.2f}"
    )


with col3:

    st.metric(
        "Payment Method",
        str(
            selected_row[
                "payment_method"
            ]
        )
    )


with col4:

    st.metric(
        "Failure Reason",
        str(
            selected_row[
                "failure_reason"
            ]
        )
    )


# --------------------------------------------------
# Run Agentic Workflow
# --------------------------------------------------

if st.button(
    "Analyze Transaction",
    type="primary",
    width="stretch"
):

    transaction = (
        selected_row
        .to_dict()
    )


    # ----------------------------------------------
    # Step 1: ML Prediction
    # ----------------------------------------------

    st.divider()

    st.subheader(
        "1. ML Recovery Prediction"
    )


    with st.spinner(
        "Running recovery prediction..."
    ):

        prediction = predict_recovery(
            transaction
        )


    probability = float(
        prediction[
            "recovery_probability"
        ]
    )


    level = prediction[
        "recovery_level"
    ]


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Recovery Probability",
            f"{probability:.2f}%"
        )


    with col2:

        st.metric(
            "Recovery Level",
            level
        )


    # ----------------------------------------------
    # Step 2: Diagnosis Agent
    # ----------------------------------------------

    st.subheader(
        "2. Diagnosis Agent"
    )


    with st.spinner(
        "Diagnosis Agent is analyzing the transaction..."
    ):

        diagnosis = diagnose_transaction(
            transaction,
            prediction
        )


    st.json(
        diagnosis
    )


    # ----------------------------------------------
    # Step 3: Decision Agent
    # ----------------------------------------------

    st.subheader(
        "3. Decision Agent"
    )


    with st.spinner(
        "Decision Agent is selecting the best strategy..."
    ):

        decision = decide_recovery_action(
            transaction,
            prediction,
            diagnosis
        )


    st.json(
        decision
    )


    # ----------------------------------------------
    # Step 4: Guardrail Engine
    # ----------------------------------------------

    st.subheader(
        "4. Guardrail Engine"
    )


    with st.spinner(
        "Validating AI decision..."
    ):

        guardrail_result = validate_action(
            transaction,
            prediction,
            decision
        )


    if guardrail_result[
        "approved"
    ]:

        st.success(
            "Action approved by guardrails."
        )

    else:

        st.warning(
            "AI action was blocked or modified "
            "by guardrails."
        )


    st.json(
        guardrail_result
    )


    # ----------------------------------------------
    # Step 5: Action Executor
    # ----------------------------------------------

    st.subheader(
        "5. Action Executor"
    )


    final_action = (
        guardrail_result[
            "final_action"
        ]
    )


    with st.spinner(
        "Executing recovery action..."
    ):

        action_result = execute_action(
            transaction,
            final_action
        )


    if action_result[
        "success"
    ]:

        st.success(
            "Recovery action executed successfully."
        )

    else:

        st.error(
            "Recovery action failed."
        )


    st.json(
        action_result
    )


    # ----------------------------------------------
    # Final Result
    # ----------------------------------------------

    st.divider()

    st.subheader(
        "RecoverX Final Decision"
    )


    final_col1, final_col2 = (
        st.columns(2)
    )


    with final_col1:

        st.metric(
            "Final Action",
            final_action
        )


    with final_col2:

        status = (
            "SUCCESS"
            if action_result[
                "success"
            ]
            else
            "FAILED"
        )


        st.metric(
            "Execution Status",
            status
        )


    # Final explanation

    st.info(
        "RecoverX completed the workflow: "
        "ML prediction → AI diagnosis → "
        "AI decision → deterministic guardrails "
        "→ recovery action."
    )