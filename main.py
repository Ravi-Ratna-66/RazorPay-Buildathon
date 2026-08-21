import json
import pandas as pd

from ml.predict import predict_recovery
from agents.diagnosis_agent import diagnose_transaction
from agents.decision_agent import decide_recovery_action
from agents.guardrails import validate_action
from agents.action_executor import execute_action


def run_recoverx(transaction):

    print("\n")
    print("=" * 60)
    print("                 RECOVERX")
    print("        Autonomous Revenue Recovery")
    print("=" * 60)

    print("\nTransaction")
    print("-" * 60)

    print(
        f"Transaction ID: "
        f"{transaction.get('transaction_id')}"
    )

    print(
        f"Customer ID: "
        f"{transaction.get('customer_id')}"
    )

    print(
        f"Amount: "
        f"₹{transaction.get('amount'):,.2f}"
    )

    print(
        f"Failure reason: "
        f"{transaction.get('failure_reason')}"
    )

    # Step 1: ML prediction
    print("\n[1] ML Recovery Prediction")
    print("-" * 60)

    prediction = predict_recovery(
        transaction
    )

    print(
        f"Recovery probability: "
        f"{prediction['recovery_probability']}%"
    )

    print(
        f"Recovery level: "
        f"{prediction['recovery_level']}"
    )

    # Step 2: Diagnosis Agent
    print("\n[2] AI Diagnosis Agent")
    print("-" * 60)

    diagnosis = diagnose_transaction(
        transaction,
        prediction
    )

    print(
        json.dumps(
            diagnosis,
            indent=4
        )
    )

    # Step 3: Decision Agent
    print("\n[3] AI Decision Agent")
    print("-" * 60)

    decision = decide_recovery_action(
        transaction,
        prediction,
        diagnosis
    )

    print(
        json.dumps(
            decision,
            indent=4
        )
    )

    # Step 4: Guardrail Engine
    print("\n[4] Guardrail Engine")
    print("-" * 60)

    guardrail_result = validate_action(
        transaction,
        prediction,
        decision
    )

    print(
        json.dumps(
            guardrail_result,
            indent=4
        )
    )

    # Step 5: Execute action
    print("\n[5] Action Executor")
    print("-" * 60)

    final_action = guardrail_result[
        "final_action"
    ]

    action_result = execute_action(
        transaction,
        final_action
    )

    print(
        json.dumps(
            action_result,
            indent=4
        )
    )

    # Final summary
    print("\n")
    print("=" * 60)
    print("              WORKFLOW COMPLETE")
    print("=" * 60)

    print(
        f"\nFinal action: "
        f"{action_result.get('action')}"
    )

    print(
        f"Action successful: "
        f"{action_result.get('success')}"
    )

    print("=" * 60)

    return {
        "transaction": transaction,
        "prediction": prediction,
        "diagnosis": diagnosis,
        "decision": decision,
        "guardrail": guardrail_result,
        "action": action_result
    }


if __name__ == "__main__":

    # Load the synthetic transaction dataset
    df = pd.read_csv(
        "data/transactions.csv"
    )

    # Select a failed transaction that has
    # not exceeded the recovery attempt limit
    candidates = df[
        (
            df["payment_status"] == "failed"
        )
        &
        (
            df["recovery_attempts"] < 3
        )
        &
        (
            df["contact_allowed"] == True
        )
    ]

    # Select the first suitable transaction
    transaction = (
        candidates
        .iloc[0]
        .to_dict()
    )

    # Run the complete RecoverX workflow
    run_recoverx(
        transaction
    )