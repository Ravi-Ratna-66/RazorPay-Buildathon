from agents.guardrails import validate_action


def create_transaction(
    failure_reason="network_error",
    recovery_attempts=0,
    contact_allowed=True
):

    return {
        "transaction_id": "TEST001",
        "customer_id": "C001",
        "amount": 500.0,
        "payment_method": "UPI",
        "payment_status": "failed",
        "failure_reason": failure_reason,
        "recovery_attempts": recovery_attempts,
        "contact_allowed": contact_allowed
    }


def create_prediction(
    probability=70.0,
    level="HIGH"
):

    return {
        "recovery_probability": probability,
        "recovery_level": level
    }


def create_decision(
    action="RETRY",
    confidence=0.85
):

    return {
        "action": action,
        "confidence": confidence,
        "reason": "Test decision"
    }


def test_valid_retry_action():

    transaction = create_transaction(
        failure_reason="network_error"
    )

    prediction = create_prediction()

    decision = create_decision(
        action="RETRY"
    )

    result = validate_action(
        transaction,
        prediction,
        decision
    )

    assert "approved" in result
    assert "final_action" in result
    assert "reasons" in result


def test_contact_not_allowed():

    transaction = create_transaction(
        contact_allowed=False
    )

    prediction = create_prediction()

    decision = create_decision(
        action="REMINDER"
    )

    result = validate_action(
        transaction,
        prediction,
        decision
    )

    assert result["final_action"] == "DO_NOT_CONTACT"


def test_too_many_recovery_attempts():

    transaction = create_transaction(
        recovery_attempts=3
    )

    prediction = create_prediction()

    decision = create_decision(
        action="RETRY"
    )

    result = validate_action(
        transaction,
        prediction,
        decision
    )

    assert result["final_action"] == "ESCALATE"