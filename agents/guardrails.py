ALLOWED_ACTIONS = {
    "RETRY",
    "ALTERNATIVE_PAYMENT",
    "REMINDER",
    "DISCOUNT",
    "ESCALATE",
    "DO_NOT_CONTACT"
}


def validate_action(
    transaction,
    recovery_prediction,
    decision
):
    """
    Validate an AI-generated recovery decision
    using deterministic business rules.
    """

    action = decision.get(
        "action",
        "ESCALATE"
    )

    reasons = []

    # Check whether the action is valid
    if action not in ALLOWED_ACTIONS:

        return {
            "approved": False,
            "final_action": "ESCALATE",
            "reasons": [
                "Invalid action returned by AI."
            ]
        }

    # Check customer contact permission
    if not transaction.get(
        "contact_allowed",
        False
    ):

        return {
            "approved": False,
            "final_action": "DO_NOT_CONTACT",
            "reasons": [
                "Customer contact permission is disabled."
            ]
        }

    # Check recovery attempts
    recovery_attempts = transaction.get(
        "recovery_attempts",
        0
    )

    if recovery_attempts >= 3:

        return {
            "approved": False,
            "final_action": "ESCALATE",
            "reasons": [
                "Maximum recovery attempts reached."
            ]
        }

    # Check transaction amount
    amount = transaction.get(
        "amount",
        0
    )

    if amount > 50000:

        if action not in [
            "ESCALATE",
            "DO_NOT_CONTACT"
        ]:

            return {
                "approved": False,
                "final_action": "ESCALATE",
                "reasons": [
                    "Transaction amount exceeds "
                    "the automatic recovery limit."
                ]
            }

    # Check discount policy
    if action == "DISCOUNT":

        if amount > 20000:

            return {
                "approved": False,
                "final_action": "ESCALATE",
                "reasons": [
                    "Automatic discounts are not "
                    "allowed above ₹20,000."
                ]
            }

    # Check ML recovery probability
    probability = recovery_prediction.get(
        "recovery_probability",
        0
    )

    if probability < 30:

        if action not in [
            "ESCALATE",
            "DO_NOT_CONTACT"
        ]:

            return {
                "approved": False,
                "final_action": "ESCALATE",
                "reasons": [
                    "ML recovery probability "
                    "is below 30%."
                ]
            }

    reasons.append(
        "Action passed all guardrail checks."
    )

    return {
        "approved": True,
        "final_action": action,
        "reasons": reasons
    }


if __name__ == "__main__":

    test_transaction = {
        "amount": 4999,
        "failure_reason": "bank_declined",
        "recovery_attempts": 0,
        "contact_allowed": True
    }

    test_prediction = {
        "recovery_probability": 73.81,
        "recovery_level": "HIGH"
    }

    test_decision = {
        "action": "ALTERNATIVE_PAYMENT",
        "confidence": 0.92,
        "reason": "Strong recovery potential."
    }

    result = validate_action(
        test_transaction,
        test_prediction,
        test_decision
    )

    print("\nRecoverX Guardrail Engine")
    print("=========================")

    print(result)