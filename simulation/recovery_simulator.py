import pandas as pd
import numpy as np
import joblib


DATA_PATH = "data/transactions.csv"
MODEL_PATH = "ml/recovery_model.pkl"


def determine_action(
    probability,
    failure_reason,
    recovery_attempts,
    amount,
    contact_allowed
):
    """
    Determine a safe recovery action for large-scale simulation.
    """

    if not contact_allowed:
        return "DO_NOT_CONTACT"

    if recovery_attempts >= 3:
        return "ESCALATE"

    if amount > 50000:
        return "ESCALATE"

    if probability < 0.30:
        return "ESCALATE"

    if failure_reason == "network_error":

        if probability >= 0.60:
            return "RETRY"

        return "ALTERNATIVE_PAYMENT"

    if failure_reason == "bank_declined":

        if probability >= 0.60:
            return "ALTERNATIVE_PAYMENT"

        return "RETRY"

    if failure_reason == "authentication_failed":
        return "ALTERNATIVE_PAYMENT"

    if failure_reason == "checkout_abandoned":

        if probability >= 0.60:
            return "REMINDER"

        return "ESCALATE"

    if failure_reason == "insufficient_funds":

        if probability >= 0.60:
            return "REMINDER"

        return "ESCALATE"

    if failure_reason == "limit_exceeded":
        return "ALTERNATIVE_PAYMENT"

    if failure_reason == "card_expired":
        return "ALTERNATIVE_PAYMENT"

    return "ESCALATE"


def simulate_outcome(
    action,
    probability,
    transaction
):
    """
    Simulate whether a recovery action succeeds.
    """

    amount = float(
        transaction["amount"]
    )

    failure_reason = transaction[
        "failure_reason"
    ]

    recovery_attempts = int(
        transaction["recovery_attempts"]
    )

    if action == "DO_NOT_CONTACT":

        success_probability = 0.0

    elif action == "ESCALATE":

        success_probability = (
            probability * 0.60
        )

    elif action == "RETRY":

        success_probability = (
            probability * 0.90
        )

        if failure_reason == "network_error":
            success_probability += 0.10

    elif action == "ALTERNATIVE_PAYMENT":

        success_probability = (
            probability * 0.95
        )

        if failure_reason in [
            "bank_declined",
            "authentication_failed",
            "limit_exceeded",
            "card_expired"
        ]:
            success_probability += 0.10

    elif action == "REMINDER":

        success_probability = (
            probability * 0.85
        )

        if failure_reason == "checkout_abandoned":
            success_probability += 0.10

    elif action == "DISCOUNT":

        success_probability = (
            probability * 1.05
        )

    else:

        success_probability = 0.0

    #Repeated attempts reduce recovery effectiveness
    success_probability -= (
        recovery_attempts * 0.05
    )

    success_probability = np.clip(
        success_probability,
        0.0,
        0.95
    )

    recovered = (
        np.random.random()
        < success_probability
    )

    if recovered:
        recovered_revenue = amount
    else:
        recovered_revenue = 0.0

    return (
        bool(recovered),
        float(recovered_revenue),
        float(success_probability)
    )


def run_simulation():
    np.random.seed(42)
    print("\n")
    print("=" * 70)
    print("                 RECOVERX SIMULATOR")
    print("=" * 70)

    #Load dataset
    df = pd.read_csv(
        DATA_PATH
    )

    #Keep only failed and abandoned transactions
    df = df[
        df["payment_status"].isin([
            "failed",
            "abandoned"
        ])
    ].copy()

    # Remove the original outcome columns because the simulator will generate its own outcomes
    original_columns = [
        "recovered",
        "recovered_revenue"
    ]

    for column in original_columns:

        if column in df.columns:
            df = df.drop(
                columns=[column]
            )

    print(
        f"\nTransactions to process: "
        f"{len(df):,}"
    )

    #Load trained ML model
    model = joblib.load(
        MODEL_PATH
    )

    features = [
        "amount",
        "transaction_type",
        "payment_method",
        "failure_reason",
        "customer_age",
        "customer_tenure_years",
        "previous_successful_payments",
        "previous_failed_payments",
        "customer_lifetime_value",
        "subscription_status",
        "cart_value",
        "time_since_failure_hours",
        "contact_allowed"
    ]

    X = df[features]

    #Predict recovery probabilities
    probabilities = model.predict_proba(
        X
    )[:, 1]

    df["predicted_probability"] = (
        probabilities
    )

    df["predicted_probability_percentage"] = (
        probabilities * 100
    )

    #Determine recovery actions
    df["recommended_action"] = df.apply(
        lambda row: determine_action(
            float(row["predicted_probability"]),
            row["failure_reason"],
            int(row["recovery_attempts"]),
            float(row["amount"]),
            bool(row["contact_allowed"])
        ),
        axis=1
    )

    #Simulate recovery outcomes
    recovered_results = []
    recovered_revenue_results = []
    success_probability_results = []

    for _, row in df.iterrows():

        (
            recovered,
            recovered_revenue,
            success_probability
        ) = simulate_outcome(
            row["recommended_action"],
            float(row["predicted_probability"]),
            row
        )

        recovered_results.append(
            recovered
        )

        recovered_revenue_results.append(
            recovered_revenue
        )

        success_probability_results.append(
            success_probability
        )

    #Store simulator results
    df["simulated_recovered"] = (
        recovered_results
    )

    df["simulated_recovered_revenue"] = (
        recovered_revenue_results
    )

    df["action_success_probability"] = (
        success_probability_results
    )

    #Calculate metrics as scalar values
    total_transactions = int(
        len(df)
    )

    total_revenue_at_risk = float(
        df["amount"].sum()
    )

    recovered_transactions = int(
        df["simulated_recovered"].sum()
    )

    recovered_revenue = float(
        df["simulated_recovered_revenue"].sum()
    )

    recovery_rate = float(
        recovered_transactions
        / total_transactions
        * 100
    )

    if total_revenue_at_risk > 0:

        revenue_recovery_rate = float(
            recovered_revenue
            / total_revenue_at_risk
            * 100
        )

    else:

        revenue_recovery_rate = 0.0

    #Action distribution
    action_counts = (
        df["recommended_action"]
        .value_counts()
    )

    #Print results
    print("\nSimulation Results")
    print("-" * 70)

    print(
        f"Total revenue at risk: "
        f"₹{total_revenue_at_risk:,.2f}"
    )

    print(
        f"Recovered revenue: "
        f"₹{recovered_revenue:,.2f}"
    )

    print(
        f"Revenue recovery rate: "
        f"{revenue_recovery_rate:.2f}%"
    )

    print(
        f"Recovered transactions: "
        f"{recovered_transactions:,}"
    )

    print(
        f"Transaction recovery rate: "
        f"{recovery_rate:.2f}%"
    )

    print("\nAction Distribution")
    print("-" * 70)

    print(
        action_counts
    )

    #Save results
    output_path = (
        "data/recovery_simulation.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSimulation results saved to: "
        f"{output_path}"
    )

    print("\n")
    print("=" * 70)
    print("             SIMULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    run_simulation()