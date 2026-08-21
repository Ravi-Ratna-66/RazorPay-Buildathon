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


VALID_ACTIONS = [
    "RETRY",
    "ALTERNATIVE_PAYMENT",
    "REMINDER",
    "DISCOUNT",
    "ESCALATE",
    "DO_NOT_CONTACT"
]


def clean_json_response(content):

    if not content:
        return None

    content = content.strip()

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


    try:

        return json.loads(
            content
        )

    except json.JSONDecodeError:

        pass


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


def deterministic_decision(
    transaction,
    recovery_prediction,
    diagnosis
):

    failure_reason = transaction.get(
        "failure_reason",
        "unknown"
    )

    payment_status = transaction.get(
        "payment_status",
        "unknown"
    )

    payment_method = transaction.get(
        "payment_method",
        "unknown"
    )

    recovery_attempts = int(
        transaction.get(
            "recovery_attempts",
            0
        )
    )

    contact_allowed = transaction.get(
        "contact_allowed",
        False
    )

    probability = float(
        recovery_prediction.get(
            "recovery_probability",
            0
        )
    )


    recommended_strategy = diagnosis.get(
        "recommended_strategy",
        "ESCALATE"
    )


    recovery_potential = diagnosis.get(
        "recovery_potential",
        "LOW"
    )


    # Safety rule: customer cannot be contacted

    if not contact_allowed:

        return {
            "action": "DO_NOT_CONTACT",
            "confidence": 1.0,
            "reason":
                "Customer contact permission "
                "is not available."
        }


    # Safety rule: too many recovery attempts

    if recovery_attempts >= 3:

        return {
            "action": "ESCALATE",
            "confidence": 0.98,
            "reason":
                "The transaction has already "
                "reached the maximum number "
                "of recovery attempts."
        }


    # Abandoned checkout

    if payment_status == "abandoned":

        return {
            "action": "REMINDER",
            "confidence": 0.90,
            "reason":
                "The checkout was abandoned, "
                "so a payment reminder is "
                "more appropriate than a retry."
        }


    # Network errors are usually temporary

    if failure_reason == "network_error":

        if probability >= 60:

            action = "RETRY"

        elif probability >= 30:

            action = "RETRY"

        else:

            action = "ESCALATE"


        return {
            "action": action,
            "confidence": 0.88,
            "reason":
                "Network errors can be temporary, "
                "making a retry appropriate when "
                "recovery probability is sufficient."
        }


    # Bank decline

    if failure_reason == "bank_declined":

        return {
            "action":
                "ALTERNATIVE_PAYMENT",

            "confidence": 0.90,

            "reason":
                "The bank declined the payment, "
                "so an alternative payment method "
                "is preferable to repeatedly retrying "
                "the same method."
        }


    # Authentication failure

    if failure_reason == "authentication_failed":

        return {
            "action":
                "ALTERNATIVE_PAYMENT",

            "confidence": 0.86,

            "reason":
                "Authentication failed, so using "
                "an alternative payment method "
                "can avoid repeating the same "
                "authentication failure."
        }


    # Insufficient funds

    if failure_reason == "insufficient_funds":

        return {
            "action":
                "REMINDER",

            "confidence": 0.82,

            "reason":
                "Insufficient funds suggest that "
                "an immediate retry may fail again. "
                "A reminder allows the customer to "
                "complete payment later."
        }


    # Expired card

    if failure_reason == "card_expired":

        return {
            "action":
                "ALTERNATIVE_PAYMENT",

            "confidence": 0.92,

            "reason":
                "The card has expired, so retrying "
                "the same payment method is unlikely "
                "to succeed."
        }


    # Limit exceeded

    if failure_reason == "limit_exceeded":

        return {
            "action":
                "ALTERNATIVE_PAYMENT",

            "confidence": 0.88,

            "reason":
                "The payment limit was exceeded, "
                "so another payment method is "
                "more appropriate."
        }


    # If diagnosis produced a valid strategy,
    # use it as a fallback.

    if recommended_strategy in VALID_ACTIONS:

        return {
            "action":
                recommended_strategy,

            "confidence":
                0.75,

            "reason":
                "The decision uses the strategy "
                "recommended by the Diagnosis Agent."
        }


    # Recovery probability is very low

    if probability < 30:

        return {
            "action":
                "ESCALATE",

            "confidence":
                0.80,

            "reason":
                "The ML model predicts a low "
                "probability of recovery."
        }


    # Final fallback

    return {
        "action":
            "ESCALATE",

        "confidence":
            0.70,

        "reason":
            "The transaction could not be "
            "matched to a safe automated "
            "recovery strategy."
    }


def decide_recovery_action(
    transaction,
    recovery_prediction,
    diagnosis
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

    diagnosis_data = json.dumps(
        diagnosis,
        indent=2
    )


    system_prompt = """
You are the Decision Agent for RecoverX.

Your job is to select the best recovery
action for a failed payment or abandoned
checkout.

You receive:

1. Transaction information
2. ML recovery prediction
3. Diagnosis from another AI agent

Available actions:

RETRY
ALTERNATIVE_PAYMENT
REMINDER
DISCOUNT
ESCALATE
DO_NOT_CONTACT

Important rules:

- Do not invent transaction information.
- Respect contact permission.
- Do not repeatedly retry transactions.
- Abandoned checkouts should generally
  receive a REMINDER.
- Network errors can generally use RETRY.
- Bank declines should generally use
  ALTERNATIVE_PAYMENT.
- Expired cards should generally use
  ALTERNATIVE_PAYMENT.
- Insufficient funds should generally use
  REMINDER.
- If contact is not allowed, use
  DO_NOT_CONTACT.
- If recovery attempts are already high,
  use ESCALATE.

Return ONLY valid JSON.

Use exactly this structure:

{
    "action": "RETRY | ALTERNATIVE_PAYMENT | REMINDER | DISCOUNT | ESCALATE | DO_NOT_CONTACT",
    "confidence": 0.0,
    "reason": "short explanation"
}

Do not include markdown.
Do not include text outside the JSON object.
"""


    user_prompt = f"""
Make a recovery decision for this transaction.

Transaction:

{transaction_data}

ML Recovery Prediction:

{prediction_data}

Diagnosis Agent Result:

{diagnosis_data}

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

            max_tokens=1000

        )


        content = (
            response
            .choices[0]
            .message
            .content
        )


        decision = clean_json_response(
            content
        )


        if decision:

            action = decision.get(
                "action"
            )

            confidence = decision.get(
                "confidence"
            )

            reason = decision.get(
                "reason"
            )


            # Validate the LLM response

            if (
                action in VALID_ACTIONS
                and isinstance(
                    confidence,
                    (int, float)
                )
                and reason
            ):

                return {
                    "action":
                        action,

                    "confidence":
                        float(confidence),

                    "reason":
                        str(reason)
                }


    except Exception as error:

        print(
            "Decision Agent error:",
            error
        )


    print(
        "Decision Agent returned "
        "invalid output. "
        "Using deterministic fallback."
    )


    return deterministic_decision(
        transaction,
        recovery_prediction,
        diagnosis
    )


# Test the Decision Agent

if __name__ == "__main__":

    test_transaction = {

        "amount": 1654.08,

        "transaction_type":
            "checkout",

        "payment_method":
            "Wallet",

        "payment_status":
            "abandoned",

        "failure_reason":
            "checkout_abandoned",

        "customer_age":
            31,

        "customer_tenure_years":
            2.8,

        "previous_successful_payments":
            8,

        "previous_failed_payments":
            1,

        "customer_lifetime_value":
            52000,

        "subscription_status":
            "not_applicable",

        "cart_value":
            1654.08,

        "time_since_failure_hours":
            44,

        "recovery_attempts":
            0,

        "contact_allowed":
            True
    }


    test_prediction = {

        "recovery_probability":
            44.41,

        "recovery_level":
            "MEDIUM"
    }


    test_diagnosis = {

        "diagnosis":
            "Checkout was abandoned using "
            "a wallet; customer has high LTV "
            "and low failure history.",

        "recovery_potential":
            "MEDIUM",

        "key_factors": [

            "checkout_abandoned",

            "high customer lifetime value",

            "44 hours since failure"

        ],

        "recommended_strategy":
            "REMINDER"
    }


    result = decide_recovery_action(

        test_transaction,

        test_prediction,

        test_diagnosis

    )


    print(
        "\nRecoverX Decision Agent"
    )

    print(
        json.dumps(
            result,
            indent=4
        )
    )