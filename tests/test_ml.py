from ml.predict import predict_recovery


def create_transaction():

    return {
        "amount": 1500.0,
        "transaction_type": "payment",
        "payment_method": "UPI",
        "payment_status": "failed",
        "failure_reason": "network_error",
        "customer_age": 28,
        "customer_tenure_years": 3.0,
        "previous_successful_payments": 10,
        "previous_failed_payments": 1,
        "customer_lifetime_value": 50000,
        "subscription_status": "not_applicable",
        "cart_value": 0,
        "time_since_failure_hours": 2,
        "recovery_attempts": 0,
        "contact_allowed": True
    }


def test_prediction_returns_probability():

    transaction = create_transaction()

    result = predict_recovery(
        transaction
    )

    assert "recovery_probability" in result
    assert "recovery_level" in result

    assert 0 <= result[
        "recovery_probability"
    ] <= 100


def test_prediction_returns_valid_level():

    transaction = create_transaction()

    result = predict_recovery(
        transaction
    )

    valid_levels = [
        "VERY HIGH",
        "HIGH",
        "MEDIUM",
        "LOW"
    ]

    assert result[
        "recovery_level"
    ] in valid_levels