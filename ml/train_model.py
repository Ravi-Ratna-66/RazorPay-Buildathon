import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# Load the dataset
df = pd.read_csv("data/transactions.csv")

# Keep only failed and abandoned transactions
df = df[
    df["payment_status"].isin([
        "failed",
        "abandoned"
    ])
].copy()

# Features available when RecoverX analyzes a transaction
features = [
    "amount",
    "transaction_type",
    "payment_method",
    "failure_reason",
    "customer_age",
    "customer_tenure_years",
    "previous_successful_payments",
    "previous_failed_payments",
    "customer_lifetime_value",
    "subscription_status",
    "cart_value",
    "time_since_failure_hours",
    "contact_allowed"
]

target = "recovered"

X = df[features]
y = df[target]

# Separate categorical and numerical features
categorical_features = [
    "transaction_type",
    "payment_method",
    "failure_reason",
    "subscription_status"
]

numerical_features = [
    "amount",
    "customer_age",
    "customer_tenure_years",
    "previous_successful_payments",
    "previous_failed_payments",
    "customer_lifetime_value",
    "cart_value",
    "time_since_failure_hours"
]

# Preprocessing for categorical features
categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

# Preprocessing for numerical features
numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    ),
    (
        "numerical",
        numerical_pipeline,
        numerical_features
    )
])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train Logistic Regression
logistic_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        LogisticRegression(
            max_iter=3000,
            C=1.0
        )
    )
])

print("\nTraining Logistic Regression...")

logistic_model.fit(
    X_train,
    y_train
)

logistic_predictions = logistic_model.predict(
    X_test
)

logistic_probabilities = logistic_model.predict_proba(
    X_test
)[:, 1]

logistic_auc = roc_auc_score(
    y_test,
    logistic_probabilities
)

print("\nLogistic Regression Results")

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            logistic_predictions
        ),
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(
            y_test,
            logistic_predictions
        ),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(
            y_test,
            logistic_predictions
        ),
        4
    )
)

print(
    "F1 Score:",
    round(
        f1_score(
            y_test,
            logistic_predictions
        ),
        4
    )
)

print(
    "ROC-AUC:",
    round(
        logistic_auc,
        4
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        logistic_predictions
    )
)


# Train Random Forest
random_forest_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced"
        )
    )
])

print("\nTraining Random Forest...")

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_predictions = (
    random_forest_model.predict(X_test)
)

random_forest_probabilities = (
    random_forest_model.predict_proba(X_test)[:, 1]
)

random_forest_auc = roc_auc_score(
    y_test,
    random_forest_probabilities
)

print("\nRandom Forest Results")

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            random_forest_predictions
        ),
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(
            y_test,
            random_forest_predictions
        ),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(
            y_test,
            random_forest_predictions
        ),
        4
    )
)

print(
    "F1 Score:",
    round(
        f1_score(
            y_test,
            random_forest_predictions
        ),
        4
    )
)

print(
    "ROC-AUC:",
    round(
        random_forest_auc,
        4
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        random_forest_predictions
    )
)


# Select the best model
if random_forest_auc > logistic_auc:

    best_model = random_forest_model
    best_model_name = "Random Forest"
    best_auc = random_forest_auc

else:

    best_model = logistic_model
    best_model_name = "Logistic Regression"
    best_auc = logistic_auc


# Save the best model
joblib.dump(
    best_model,
    "ml/recovery_model.pkl"
)

print("\nBest model:", best_model_name)

print(
    "Best ROC-AUC:",
    round(best_auc, 4)
)

print(
    "\nModel saved to:"
    " ml/recovery_model.pkl"
)