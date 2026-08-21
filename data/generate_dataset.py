import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

#Dataset configuration
NUM_TRANSACTIONS = 50000
NUM_CUSTOMERS = 10000

np.random.seed(42)
random.seed(42)

fake = Faker()
Faker.seed(42)

PAYMENT_METHODS = [
    "UPI",
    "Credit Card",
    "Debit Card",
    "Net Banking",
    "Wallet"
]

TRANSACTION_TYPES = [
    "payment",
    "subscription",
    "checkout"
]

FAILURE_REASONS = [
    "insufficient_funds",
    "bank_declined",
    "card_expired",
    "network_error",
    "authentication_failed",
    "limit_exceeded"
]

#Create customer profiles
customers = []

for i in range(NUM_CUSTOMERS):

    customer_id = f"C{i + 10001}"

    customer_age = random.randint(18, 60)

    customer_tenure = round(
        np.random.uniform(0.1, 5.0),
        2
    )

    previous_successful_payments = max(
        0,
        int(
            np.random.poisson(
                2 + customer_tenure * 3
            )
        )
    )

    previous_failed_payments = max(
        0,
        int(np.random.poisson(1.5))
    )

    customer_lifetime_value = round(
        max(
            500,
            previous_successful_payments
            * np.random.uniform(700, 4000)
        ),
        2
    )

    preferred_payment_method = random.choice(
        PAYMENT_METHODS
    )

    customers.append({
        "customer_id": customer_id,
        "customer_age": customer_age,
        "customer_tenure_years": customer_tenure,
        "previous_successful_payments":
            previous_successful_payments,
        "previous_failed_payments":
            previous_failed_payments,
        "customer_lifetime_value":
            customer_lifetime_value,
        "preferred_payment_method":
            preferred_payment_method
    })

customers_df = pd.DataFrame(customers)

#Generate transaction records
records = []

start_date = datetime.now() - timedelta(days=365)

for i in range(NUM_TRANSACTIONS):

    customer = customers_df.sample(
        n=1
    ).iloc[0]

    transaction_id = f"T{i + 1:06d}"

    transaction_date = (
        start_date
        + timedelta(
            minutes=random.randint(
                0,
                525600
            )
        )
    )

    transaction_type = random.choices(
        TRANSACTION_TYPES,
        weights=[0.60, 0.25, 0.15],
        k=1
    )[0]

    #Generate transaction amount
    amount = round(
        np.random.lognormal(
            mean=7.5,
            sigma=0.85
        ),
        2
    )

    amount = min(
        max(amount, 100),
        100000
    )

    #Select payment method
    if random.random() < 0.80:
        payment_method = customer[
            "preferred_payment_method"
        ]
    else:
        payment_method = random.choice(
            PAYMENT_METHODS
        )

    #Generate payment status
    payment_status = random.choices(
        ["success", "failed", "abandoned"],
        weights=[0.78, 0.17, 0.05],
        k=1
    )[0]

    #Generate failure reason
    if payment_status == "failed":

        failure_reason = random.choices(
            FAILURE_REASONS,
            weights=[
                0.15,
                0.20,
                0.15,
                0.20,
                0.15,
                0.15
            ],
            k=1
        )[0]

    elif payment_status == "abandoned":

        failure_reason = "checkout_abandoned"

    else:

        failure_reason = "none"

    #Generate subscription status
    if transaction_type == "subscription":

        if payment_status == "failed":
            subscription_status = "renewal_failed"

        else:
            subscription_status = random.choice([
                "active",
                "cancelled"
            ])

    else:

        subscription_status = "not_applicable"

    #Generate cart value
    if transaction_type == "checkout":

        cart_value = round(
            amount * np.random.uniform(
                1.0,
                1.5
            ),
            2
        )

    else:

        cart_value = 0

    #Time since failure
    if payment_status in [
        "failed",
        "abandoned"
    ]:

        time_since_failure_hours = random.randint(
            1,
            72
        )

    else:

        time_since_failure_hours = 0

    #Previous recovery attempts
    if payment_status in [
        "failed",
        "abandoned"
    ]:

        recovery_attempts = random.choices(
            [0, 1, 2, 3],
            weights=[
                0.50,
                0.25,
                0.15,
                0.10
            ],
            k=1
        )[0]

    else:

        recovery_attempts = 0

    #Customer contact permission
    contact_allowed = random.choices(
        [True, False],
        weights=[0.94, 0.06],
        k=1
    )[0]

    #Calculate recovery probability
    if payment_status in [
        "failed",
        "abandoned"
    ] and contact_allowed:

        recovery_score = -1.5

        #Customer history
        recovery_score += (
            min(
                customer[
                    "previous_successful_payments"
                ],
                20
            ) * 0.12
        )

        recovery_score -= (
            min(
                customer[
                    "previous_failed_payments"
                ],
                8
            ) * 0.15
        )

        #Customer loyalty
        recovery_score += (
            customer[
                "customer_tenure_years"
            ] * 0.20
        )

        #Failure reason
        if failure_reason == "network_error":
            recovery_score += 0.80

        elif failure_reason == "bank_declined":
            recovery_score += 0.45

        elif failure_reason == "authentication_failed":
            recovery_score += 0.10

        elif failure_reason == "limit_exceeded":
            recovery_score -= 0.20

        elif failure_reason == "insufficient_funds":
            recovery_score -= 0.45

        elif failure_reason == "card_expired":
            recovery_score -= 0.60

        elif failure_reason == "checkout_abandoned":
            recovery_score += 0.25

        #Recovery attempts
        recovery_score -= (
            recovery_attempts * 0.35
        )

        #Time since failure
        if time_since_failure_hours <= 6:
            recovery_score += 0.40

        elif time_since_failure_hours <= 24:
            recovery_score += 0.15

        else:
            recovery_score -= 0.20

        #Transaction amount
        if amount <= 5000:
            recovery_score += 0.30

        elif amount <= 20000:
            recovery_score += 0.10

        elif amount > 50000:
            recovery_score -= 0.45

        #High-value loyal customers
        if (
            customer[
                "customer_lifetime_value"
            ] > 50000
            and
            customer[
                "previous_successful_payments"
            ] >= 10
        ):

            recovery_score += 0.40

        #Convert score into probability
        recovery_probability = (
            1 /
            (
                1 +
                np.exp(
                    -recovery_score
                )
            )
        )

        #Add small random variation
        recovery_probability += np.random.normal(
            0,
            0.03
        )

        recovery_probability = np.clip(
            recovery_probability,
            0.02,
            0.95
        )

    else:

        recovery_probability = 0.0

    #Generate recovery outcome
    if payment_status in [
        "failed",
        "abandoned"
    ] and contact_allowed:

        recovered = np.random.binomial(
            1,
            recovery_probability
        )

    else:

        recovered = 0

    #Revenue at risk
    if payment_status in [
        "failed",
        "abandoned"
    ]:

        revenue_at_risk = amount

    else:

        revenue_at_risk = 0

    #Recovered revenue
    if recovered == 1:

        recovered_revenue = amount

    else:

        recovered_revenue = 0

    records.append({

        "transaction_id":
            transaction_id,

        "customer_id":
            customer["customer_id"],

        "transaction_date":
            transaction_date,

        "amount":
            amount,

        "transaction_type":
            transaction_type,

        "payment_method":
            payment_method,

        "payment_status":
            payment_status,

        "failure_reason":
            failure_reason,

        "customer_age":
            customer["customer_age"],

        "customer_tenure_years":
            customer[
                "customer_tenure_years"
            ],

        "previous_successful_payments":
            customer[
                "previous_successful_payments"
            ],

        "previous_failed_payments":
            customer[
                "previous_failed_payments"
            ],

        "customer_lifetime_value":
            customer[
                "customer_lifetime_value"
            ],

        "subscription_status":
            subscription_status,

        "cart_value":
            cart_value,

        "time_since_failure_hours":
            time_since_failure_hours,

        "recovery_attempts":
            recovery_attempts,

        "contact_allowed":
            contact_allowed,

        "revenue_at_risk":
            round(
                revenue_at_risk,
                2
            ),

        "recovered":
            recovered,

        "recovered_revenue":
            round(
                recovered_revenue,
                2
            )
    })


#Create DataFrame
df = pd.DataFrame(records)

#Sort transactions chronologically
df = df.sort_values(
    "transaction_date"
).reset_index(drop=True)

#Save the dataset
output_path = "data/transactions.csv"

df.to_csv(
    output_path,
    index=False
)

# Display summary
print("\nRecoverX Dataset Generated Successfully")

print(
    f"\nTotal transactions: "
    f"{len(df):,}"
)

print(
    f"Total customers: "
    f"{df['customer_id'].nunique():,}"
)

print(
    f"Total columns: "
    f"{len(df.columns)}"
)

print("\nPayment Status:")
print(
    df["payment_status"].value_counts()
)

print("\nFailure Reasons:")
print(
    df["failure_reason"].value_counts()
)

print("\nRecovery Results:")
print(
    df["recovered"].value_counts()
)

loss_df = df[
    df["payment_status"].isin([
        "failed",
        "abandoned"
    ])
]

print("\nRecovery Rate:")

print(
    f"{loss_df['recovered'].mean() * 100:.2f}%"
)

print("\nRevenue Metrics:")

print(
    f"Revenue at risk: "
    f"₹{df['revenue_at_risk'].sum():,.2f}"
)

print(
    f"Recovered revenue: "
    f"₹{df['recovered_revenue'].sum():,.2f}"
)

print("\nDataset Preview:")

print(
    df.head(5).to_string()
)

print(
    f"\nSaved to: {output_path}"
)