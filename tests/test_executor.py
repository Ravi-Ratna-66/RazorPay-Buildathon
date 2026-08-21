from agents.action_executor import execute_action


def create_transaction():

    return {
        "transaction_id": "TEST001",
        "customer_id": "C001",
        "amount": 500.0,
        "payment_method": "UPI",
        "payment_status": "failed",
        "failure_reason": "network_error",
        "contact_allowed": True
    }


def test_retry_execution():

    transaction = create_transaction()

    result = execute_action(
        transaction,
        "RETRY"
    )

    assert result["success"] is True
    assert result["action"] == "RETRY"
    assert "timestamp" in result


def test_reminder_execution():

    transaction = create_transaction()

    result = execute_action(
        transaction,
        "REMINDER"
    )

    assert result["success"] is True
    assert result["action"] == "REMINDER"


def test_alternative_payment_execution():

    transaction = create_transaction()

    result = execute_action(
        transaction,
        "ALTERNATIVE_PAYMENT"
    )

    assert result["success"] is True
    assert result["action"] == "ALTERNATIVE_PAYMENT"