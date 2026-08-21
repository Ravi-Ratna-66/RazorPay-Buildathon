import os
import json
import re

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


api_key = os.getenv("GROQ_API_KEY")


if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found in the .env file."
    )


client = Groq(
    api_key=api_key
)


MODEL_NAME = "openai/gpt-oss-120b"


def clean_json_response(content):

    if not content:
        return None

    content = content.strip()

    # Remove markdown code fences
    content = re.sub(
        r"^```(?:json)?\s*",
        "",
        content,
        flags=re.IGNORECASE
    )

    content = re.sub(
        r"\s*```$",
        "",
        content
    )

    content = content.strip()

    # Try direct JSON parsing
    try:

        return json.loads(
            content
        )

    except json.JSONDecodeError:

        pass

    # Try extracting JSON object
    match = re.search(
        r"\{.*\}",
        content,
        re.DOTALL
    )

    if match:

        try:

            return json.loads(
                match.group(0)
            )

        except json.JSONDecodeError:

            return None

    return None


def fallback_diagnosis(
    transaction,
    recovery_prediction
):

    failure_reason = transaction.get(
        "failure_reason",
        "unknown"
    )

    payment_method = transaction.get(
        "payment_method",
        "unknown"
    )

    probability = recovery_prediction.get(
        "recovery_probability",
        0
    )

    level = recovery_prediction.get(
        "recovery_level",
        "LOW"
    )

    attempts = transaction.get(
        "recovery_attempts",
        0
    )

    contact_allowed = transaction.get(
        "contact_allowed",
        False
    )

    if not contact_allowed:

        strategy = "DO_NOT_CONTACT"

    elif attempts >= 3:

        strategy = "ESCALATE"

    elif failure_reason == "network_error":

        strategy = "RETRY"

    elif failure_reason == "bank_declined":

        strategy = "ALTERNATIVE_PAYMENT"

    elif failure_reason == "authentication_failed":

        strategy = "ALTERNATIVE_PAYMENT"

    elif failure_reason == "checkout_abandoned":

        strategy = "REMINDER"

    elif failure_reason == "insufficient_funds":

        strategy = "REMINDER"

    elif failure_reason == "card_expired":

        strategy = "ALTERNATIVE_PAYMENT"

    elif failure_reason == "limit_exceeded":

        strategy = "ALTERNATIVE_PAYMENT"

    else:

        strategy = "ESCALATE"


    return {

        "diagnosis":
            f"Payment failed due to "
            f"{failure_reason} using "
            f"{payment_method}. "
            f"The ML model estimates a "
            f"{probability:.2f}% recovery probability.",

        "recovery_potential":
            level,

        "key_factors": [

            f"failure reason: {failure_reason}",

            f"payment method: {payment_method}",

            f"ML recovery probability: "
            f"{probability:.2f}%",

            f"recovery attempts: {attempts}",

            f"contact allowed: "
            f"{contact_allowed}"

        ],

        "recommended_strategy":
            strategy

    }


def diagnose_transaction(
    transaction,
    recovery_prediction
):

    transaction_data = json.dumps(
        transaction,
        indent=2,
        default=str
    )

    prediction_data = json.dumps(
        recovery_prediction,
        indent=2
    )


    system_prompt = """
You are the Diagnosis Agent for RecoverX.

RecoverX is an autonomous revenue recovery
system for failed payments and abandoned
checkouts.

Your responsibility is to diagnose why a
transaction may or may not be recoverable.

You receive:

1. Transaction information
2. Recovery probability from a separate
   machine learning model

The ML probability is authoritative.
Do not change it.

Analyze:

- Payment failure reason
- Payment method
- Customer payment history
- Customer tenure
- Customer lifetime value
- Transaction amount
- Time since failure
- Previous recovery attempts
- Contact permission
- ML recovery probability

Important:

Only use information present in the
transaction and ML prediction.

Do not invent facts.

Return ONLY a JSON object.

Use exactly this structure:

{
    "diagnosis": "short explanation",
    "recovery_potential": "VERY HIGH | HIGH | MEDIUM | LOW",
    "key_factors": [
        "factor 1",
        "factor 2",
        "factor 3"
    ],
    "recommended_strategy": "RETRY | ALTERNATIVE_PAYMENT | REMINDER | DISCOUNT | ESCALATE | DO_NOT_CONTACT"
}

Do not include markdown.

Do not include explanations outside JSON.
"""


    user_prompt = f"""
Analyze this RecoverX transaction.

Transaction:

{transaction_data}

Machine Learning Recovery Prediction:

{prediction_data}

Return ONLY the JSON object.
"""


    try:

        response = client.chat.completions.create(

            model=MODEL_NAME,

            messages=[

                {
                    "role": "system",
                    "content": system_prompt
                },

                {
                    "role": "user",
                    "content": user_prompt
                }

            ],

            temperature=0.1,

            max_tokens=1200

        )


        content = (
            response
            .choices[0]
            .message
            .content
        )


        diagnosis = clean_json_response(
            content
        )


        if diagnosis:

            return diagnosis


    except Exception as error:

        print(
            "Diagnosis Agent error:",
            error
        )


    print(
        "Diagnosis Agent returned "
        "an invalid or empty response. "
        "Using deterministic fallback."
    )


    return fallback_diagnosis(
        transaction,
        recovery_prediction
    )


# Test the Diagnosis Agent
if __name__ == "__main__":

    test_transaction = {

        "amount": 4999,

        "transaction_type":
            "payment",

        "payment_method":
            "UPI",

        "payment_status":
            "failed",

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

        "recovery_attempts":
            0,

        "contact_allowed":
            True
    }


    test_prediction = {

        "recovery_probability":
            73.81,

        "recovery_level":
            "HIGH"
    }


    result = diagnose_transaction(
        test_transaction,
        test_prediction
    )


    print(
        "\nRecoverX Diagnosis Agent"
    )

    print(
        json.dumps(
            result,
            indent=4
        )
    )