from datetime import datetime
from zoneinfo import ZoneInfo


def execute_action(
    transaction,
    action
):
    """
    Execute a simulated recovery action.
    """

    transaction_id = transaction.get(
        "transaction_id",
        "UNKNOWN"
    )

    customer_id = transaction.get(
        "customer_id",
        "UNKNOWN"
    )

    amount = transaction.get(
        "amount",
        0
    )
    timestamp = datetime.now( ZoneInfo("Asia/Kolkata")).isoformat()

    if action == "RETRY":

        return {
            "success": True,
            "action": "RETRY",
            "transaction_id": transaction_id,
            "message": (
                "Payment retry initiated "
                "using the existing payment method."
            ),
            "timestamp": timestamp
        }

    if action == "ALTERNATIVE_PAYMENT":

        return {
            "success": True,
            "action": "ALTERNATIVE_PAYMENT",
            "transaction_id": transaction_id,
            "message": (
                "Alternative payment option generated "
                "for the customer."
            ),
            "available_methods": [
                "UPI",
                "Credit Card",
                "Net Banking"
            ],
            "timestamp": timestamp
        }

    if action == "REMINDER":

        return {
            "success": True,
            "action": "REMINDER",
            "customer_id": customer_id,
            "message": (
                f"Payment reminder generated for "
                f"₹{amount:,.2f}."
            ),
            "timestamp": timestamp
        }

    if action == "DISCOUNT":

        return {
            "success": True,
            "action": "DISCOUNT",
            "customer_id": customer_id,
            "discount_percentage": 5,
            "message": (
                "A 5% recovery incentive has been "
                "generated."
            ),
            "timestamp": timestamp
        }

    if action == "ESCALATE":

        return {
            "success": True,
            "action": "ESCALATE",
            "transaction_id": transaction_id,
            "message": (
                "Transaction has been sent for "
                "human review."
            ),
            "timestamp": timestamp
        }

    if action == "DO_NOT_CONTACT":

        return {
            "success": True,
            "action": "DO_NOT_CONTACT",
            "customer_id": customer_id,
            "message": (
                "No customer contact will be made."
            ),
            "timestamp": timestamp
        }

    return {
        "success": False,
        "action": action,
        "message": "Unknown action.",
        "timestamp": timestamp
    }


#Test 
if __name__ == "__main__":

    test_transaction = {
        "transaction_id": "T000001",
        "customer_id": "C10001",
        "amount": 4999
    }

    test_action = "ALTERNATIVE_PAYMENT"

    result = execute_action(
        test_transaction,
        test_action
    )

    print("\nRecoverX Action Executor")

    print(result)