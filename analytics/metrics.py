import pandas as pd


SIMULATION_PATH = "data/recovery_simulation.csv"
BASELINE_PATH = "data/baseline_comparison.csv"


def load_simulation_data():
    return pd.read_csv(
        SIMULATION_PATH
    )


def calculate_overview_metrics(df):
    total_transactions = len(df)

    revenue_at_risk = float(
        df["amount"].sum()
    )

    recovered_revenue = float(
        df["simulated_recovered_revenue"].sum()
    )

    recovered_transactions = int(
        df["simulated_recovered"].sum()
    )

    if revenue_at_risk > 0:

        revenue_recovery_rate = (
            recovered_revenue
            / revenue_at_risk
            * 100
        )

    else:

        revenue_recovery_rate = 0.0

    if total_transactions > 0:

        transaction_recovery_rate = (
            recovered_transactions
            / total_transactions
            * 100
        )

    else:

        transaction_recovery_rate = 0.0

    return {
        "total_transactions": total_transactions,
        "revenue_at_risk": revenue_at_risk,
        "recovered_revenue": recovered_revenue,
        "recovered_transactions": recovered_transactions,
        "revenue_recovery_rate":
            revenue_recovery_rate,
        "transaction_recovery_rate":
            transaction_recovery_rate
    }


def calculate_action_metrics(df):
    action_counts = (
        df["recommended_action"]
        .value_counts()
        .to_dict()
    )

    action_revenue = (
        df.groupby(
            "recommended_action"
        )[
            "simulated_recovered_revenue"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
        .to_dict()
    )

    return {
        "action_counts": action_counts,
        "action_revenue": action_revenue
    }


def calculate_failure_metrics(df):
    failure_metrics = (
        df.groupby(
            "failure_reason"
        )
        .agg(
            transactions=(
                "transaction_id",
                "count"
            ),
            revenue_at_risk=(
                "amount",
                "sum"
            ),
            recovered_revenue=(
                "simulated_recovered_revenue",
                "sum"
            ),
            recovered_transactions=(
                "simulated_recovered",
                "sum"
            )
        )
        .reset_index()
    )

    failure_metrics[
        "recovery_rate"
    ] = (
        failure_metrics[
            "recovered_transactions"
        ]
        / failure_metrics[
            "transactions"
        ]
        * 100
    )

    failure_metrics[
        "revenue_recovery_rate"
    ] = (
        failure_metrics[
            "recovered_revenue"
        ]
        / failure_metrics[
            "revenue_at_risk"
        ]
        * 100
    )

    return failure_metrics


def calculate_payment_method_metrics(df):
    payment_metrics = (
        df.groupby(
            "payment_method"
        )
        .agg(
            transactions=(
                "transaction_id",
                "count"
            ),
            revenue_at_risk=(
                "amount",
                "sum"
            ),
            recovered_revenue=(
                "simulated_recovered_revenue",
                "sum"
            ),
            recovered_transactions=(
                "simulated_recovered",
                "sum"
            )
        )
        .reset_index()
    )

    payment_metrics[
        "recovery_rate"
    ] = (
        payment_metrics[
            "recovered_transactions"
        ]
        / payment_metrics[
            "transactions"
        ]
        * 100
    )

    return payment_metrics


def calculate_baseline_comparison():

    df = pd.read_csv(
        BASELINE_PATH
    )

    recoverx_revenue = float(
        df[
            "simulated_recovered_revenue"
        ].sum()
    )

    naive_revenue = float(
        df[
            "naive_recovered_revenue"
        ].sum()
    )

    baseline_revenue = float(
        df[
            "baseline_recovered_revenue"
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

    return {
        "baseline_revenue":
            baseline_revenue,

        "naive_revenue":
            naive_revenue,

        "recoverx_revenue":
            recoverx_revenue,

        "revenue_uplift":
            revenue_uplift,

        "uplift_percentage":
            uplift_percentage
    }


def print_metrics():

    df = load_simulation_data()

    overview = calculate_overview_metrics(
        df
    )

    actions = calculate_action_metrics(
        df
    )

    failures = calculate_failure_metrics(
        df
    )

    payment_methods = (
        calculate_payment_method_metrics(
            df
        )
    )

    comparison = (
        calculate_baseline_comparison()
    )

    print("\n")
    print("=" * 70)
    print("                 RECOVERX ANALYTICS")
    print("=" * 70)

    print("\nOverview")
    print("-" * 70)

    print(
        f"Transactions analyzed: "
        f"{overview['total_transactions']:,}"
    )

    print(
        f"Revenue at risk: "
        f"₹{overview['revenue_at_risk']:,.2f}"
    )

    print(
        f"Recovered revenue: "
        f"₹{overview['recovered_revenue']:,.2f}"
    )

    print(
        f"Recovered transactions: "
        f"{overview['recovered_transactions']:,}"
    )

    print(
        f"Revenue recovery rate: "
        f"{overview['revenue_recovery_rate']:.2f}%"
    )

    print(
        f"Transaction recovery rate: "
        f"{overview['transaction_recovery_rate']:.2f}%"
    )

    print("\nAction Distribution")
    print("-" * 70)

    for action, count in actions[
        "action_counts"
    ].items():

        print(
            f"{action:<25} {count:,}"
        )

    print("\nRecovery by Failure Reason")
    print("-" * 70)

    print(
        failures[
            [
                "failure_reason",
                "transactions",
                "revenue_at_risk",
                "recovered_revenue",
                "recovery_rate"
            ]
        ].to_string(
            index=False
        )
    )

    print("\nRecovery by Payment Method")
    print("-" * 70)

    print(
        payment_methods[
            [
                "payment_method",
                "transactions",
                "revenue_at_risk",
                "recovered_revenue",
                "recovery_rate"
            ]
        ].to_string(
            index=False
        )
    )

    print("\nBaseline Comparison")
    print("-" * 70)

    print(
        f"Baseline revenue: "
        f"₹{comparison['baseline_revenue']:,.2f}"
    )

    print(
        f"Naive revenue: "
        f"₹{comparison['naive_revenue']:,.2f}"
    )

    print(
        f"RecoverX revenue: "
        f"₹{comparison['recoverx_revenue']:,.2f}"
    )

    print(
        f"Revenue uplift: "
        f"₹{comparison['revenue_uplift']:,.2f}"
    )

    print(
        f"Uplift percentage: "
        f"{comparison['uplift_percentage']:.2f}%"
    )

    print("\n")
    print("=" * 70)
    print("              ANALYTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print_metrics()