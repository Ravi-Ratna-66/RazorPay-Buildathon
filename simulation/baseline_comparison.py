import pandas as pd
import numpy as np


DATA_PATH = "data/recovery_simulation.csv"


def simulate_baseline(df):
    """
    Simulate a baseline where no automated
    recovery action is performed.
    """

    df["baseline_recovered"] = False

    df["baseline_recovered_revenue"] = 0.0

    return df


def simulate_naive_strategy(df):
    """
    Simulate a naive strategy that attempts
    recovery on every eligible transaction.
    """

    recovered_results = []
    revenue_results = []

    for _, row in df.iterrows():

        amount = float(
            row["amount"]
        )

        contact_allowed = bool(
            row["contact_allowed"]
        )

        if not contact_allowed:

            recovered = False

        else:

            # Simple strategy:
            # attempt recovery on every transaction
            success_probability = 0.25

            recovered = (
                np.random.random()
                < success_probability
            )

        if recovered:

            revenue = amount

        else:

            revenue = 0.0

        recovered_results.append(
            recovered
        )

        revenue_results.append(
            revenue
        )

    df["naive_recovered"] = (
        recovered_results
    )

    df["naive_recovered_revenue"] = (
        revenue_results
    )

    return df


def main():

    print("\n")
    print("=" * 70)
    print("              RECOVERX BASELINE ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(
        DATA_PATH
    )

    total_transactions = len(df)

    total_revenue_at_risk = float(
        df["amount"].sum()
    )

    #Baseline
    df = simulate_baseline(
        df
    )

    #Naive strategy
    df = simulate_naive_strategy(
        df
    )

    #RecoverX results
    recoverx_revenue = float(
        df[
            "simulated_recovered_revenue"
        ].sum()
    )

    recoverx_transactions = int(
        df[
            "simulated_recovered"
        ].sum()
    )

    recoverx_rate = (
        recoverx_transactions
        / total_transactions
        * 100
    )

    recoverx_revenue_rate = (
        recoverx_revenue
        / total_revenue_at_risk
        * 100
    )

    #Baseline results
    baseline_revenue = float(
        df[
            "baseline_recovered_revenue"
        ].sum()
    )

    baseline_transactions = int(
        df[
            "baseline_recovered"
        ].sum()
    )

    baseline_rate = (
        baseline_transactions
        / total_transactions
        * 100
    )

    baseline_revenue_rate = (
        baseline_revenue
        / total_revenue_at_risk
        * 100
    )

    #Naive results
    naive_revenue = float(
        df[
            "naive_recovered_revenue"
        ].sum()
    )

    naive_transactions = int(
        df[
            "naive_recovered"
        ].sum()
    )

    naive_rate = (
        naive_transactions
        / total_transactions
        * 100
    )

    naive_revenue_rate = (
        naive_revenue
        / total_revenue_at_risk
        * 100
    )

    #Revenue uplift
    revenue_uplift_vs_naive = (
        recoverx_revenue
        - naive_revenue
    )

    revenue_uplift_percentage = (
        revenue_uplift_vs_naive
        / max(
            naive_revenue,
            1
        )
        * 100
    )

    print("\nComparison")
    print("-" * 70)

    print(
        f"{'Metric':<30}"
        f"{'Baseline':>15}"
        f"{'Naive':>15}"
        f"{'RecoverX':>15}"
    )

    print("-" * 70)

    print(
        f"{'Recovered Revenue':<30}"
        f"₹{baseline_revenue:>13,.0f}"
        f"₹{naive_revenue:>13,.0f}"
        f"₹{recoverx_revenue:>13,.0f}"
    )

    print(
        f"{'Revenue Recovery Rate':<30}"
        f"{baseline_revenue_rate:>14.2f}%"
        f"{naive_revenue_rate:>14.2f}%"
        f"{recoverx_revenue_rate:>14.2f}%"
    )

    print(
        f"{'Recovered Transactions':<30}"
        f"{baseline_transactions:>15,}"
        f"{naive_transactions:>15,}"
        f"{recoverx_transactions:>15,}"
    )

    print(
        f"{'Transaction Recovery Rate':<30}"
        f"{baseline_rate:>14.2f}%"
        f"{naive_rate:>14.2f}%"
        f"{recoverx_rate:>14.2f}%"
    )

    print("-" * 70)

    print(
        f"\nRecoverX revenue uplift "
        f"vs naive strategy: "
        f"₹{revenue_uplift_vs_naive:,.2f}"
    )

    print(
        f"Revenue uplift percentage: "
        f"{revenue_uplift_percentage:.2f}%"
    )

    #Save comparison dataset
    output_path = (
        "data/baseline_comparison.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nComparison data saved to: "
        f"{output_path}"
    )

    print("\n")
    print("=" * 70)
    print("            BASELINE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    main()