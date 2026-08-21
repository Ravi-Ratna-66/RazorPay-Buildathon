import pandas as pd
import matplotlib.pyplot as plt

#Load the dataset
df = pd.read_csv("data/transactions.csv")

print("Dataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nPayment status:")
print(df["payment_status"].value_counts())

#Keep only transactions where revenue may have been lost
loss_df = df[
    df["payment_status"].isin(["failed", "abandoned"])
].copy()

print("\nRevenue-loss events:")
print(len(loss_df))

#Overall recovery rate
recovery_rate = loss_df["recovered"].mean() * 100

print("\nOverall recovery rate:")
print(f"{recovery_rate:.2f}%")

#Recovery rate by failure reason
failure_recovery = (
    loss_df
    .groupby("failure_reason")["recovered"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

print("\nRecovery rate by failure reason:")
print(failure_recovery)

#Recovery rate by payment method
payment_recovery = (
    loss_df
    .groupby("payment_method")["recovered"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

print("\nRecovery rate by payment method:")
print(payment_recovery)

#Recovery rate by previous successful payments
history_recovery = (
    loss_df
    .groupby("previous_successful_payments")["recovered"]
    .mean()
    * 100
)

print("\nRecovery rate by previous successful payments:")
print(history_recovery)

#Recovery rate by number of recovery attempts
attempt_recovery = (
    loss_df
    .groupby("recovery_attempts")["recovered"]
    .mean()
    * 100
)

print("\nRecovery rate by recovery attempts:")
print(attempt_recovery)

#Recovery rate by transaction type
type_recovery = (
    loss_df
    .groupby("transaction_type")["recovered"]
    .mean()
    * 100
)

print("\nRecovery rate by transaction type:")
print(type_recovery)

#Plot recovery rate by failure reason
failure_recovery.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Recovery Rate by Failure Reason")
plt.xlabel("Failure Reason")
plt.ylabel("Recovery Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("data/recovery_by_failure_reason.png")

plt.show()

#Plot recovery rate by payment method
payment_recovery.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Recovery Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Recovery Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("data/recovery_by_payment_method.png")

plt.show()

print("\nEDA completed successfully.")