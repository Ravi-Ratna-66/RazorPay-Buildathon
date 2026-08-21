import pandas as pd
import joblib


MODEL_PATH = "ml/recovery_model.pkl"


#Load the trained model
model = joblib.load(MODEL_PATH)


def predict_recovery(transaction):
    """
    Predict the probability that a failed or abandoned
    transaction can be recovered.
    """

    #Convert input transaction into a DataFrame
    data = pd.DataFrame([transaction])

    #Get recovery probability
    probability = model.predict_proba(
        data
    )[0][1]

    probability_percentage = probability * 100

    #Assign a recovery level
    if probability >= 0.80:
        recovery_level = "VERY HIGH"

    elif probability >= 0.60:
        recovery_level = "HIGH"

    elif probability >= 0.40:
        recovery_level = "MEDIUM"

    else:
        recovery_level = "LOW"

    return {
        "recovery_probability":
            round(
                probability_percentage,
                2
            ),

        "recovery_level":
            recovery_level
    }


#Test transaction
if __name__ == "__main__":

    test_transaction = {

        "amount": 4999,

        "transaction_type":
            "payment",

        "payment_method":
            "UPI",

        "failure_reason":
            "bank_declined",

        "customer_age":
            28,

        "customer_tenure_years":
            3.2,

        "previous_successful_payments":
            12,

        "previous_failed_payments":
            1,

        "customer_lifetime_value":
            65000,

        "subscription_status":
            "not_applicable",

        "cart_value":
            0,

        "time_since_failure_hours":
            4,

        "contact_allowed":
            True
    }

    result = predict_recovery(
        test_transaction
    )

    print("\nRecoverX Prediction")

    print(
        f"Recovery Probability: "
        f"{result['recovery_probability']}%"
    )

    print(
        f"Recovery Level: "
        f"{result['recovery_level']}"
    )