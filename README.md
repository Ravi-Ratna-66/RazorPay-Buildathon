# RecoverX

## Autonomous AI-Powered Revenue Recovery

RecoverX is an AI-powered revenue recovery system designed to identify failed or abandoned payment transactions, estimate their probability of recovery, understand why the payment failed, select an appropriate recovery strategy, validate that decision using deterministic guardrails, and execute the recovery action.

The system combines machine learning, LLM-based agents, deterministic safety rules, and simulation analytics into an end-to-end autonomous recovery workflow.


## Problem

Failed payments and abandoned checkouts can result in significant revenue leakage.

A simple recovery system may apply the same action to every failed transaction, such as repeatedly retrying a payment.

However, different payment failures require different strategies.

For example:

- A temporary network error may benefit from a retry.
- A bank decline may require an alternative payment method.
- An abandoned checkout may require a reminder.
- An expired card should not repeatedly retry the same card.
- A customer who has opted out of communication should not be contacted.

RecoverX addresses this problem by combining predictive modeling with contextual AI reasoning and deterministic safety controls.


## Solution

RecoverX follows an autonomous recovery pipeline:

Failed / Abandoned Transaction
              |
              v
       ML Recovery Model
              |
              v
       Diagnosis Agent
              |
              v
        Decision Agent
              |
              v
       Guardrail Engine
              |
              v
        Action Executor
              |
              v
        Recovery Action


        1. Synthetic Dataset

I created a synthetic dataset containing 50,000 transactions.

The dataset contains information such as:

Transaction amount
Payment method
Payment status
Failure reason
Customer age
Customer tenure
Previous successful payments
Previous failed payments
Customer lifetime value
Recovery attempts
Time since failure
Contact permission
Revenue at risk
Recovery result

I used Faker and Python to generate the dataset.

To generate the dataset:

python data/generate_dataset.py
2. Data Analysis

Before training the model, I analyzed the dataset to understand the payment failures and recovery patterns.

I looked at things like:

Which payment failures happen most often
Which failure reasons have better recovery rates
Which payment methods recover better
How previous payment history affects recovery
How recovery attempts affect the result

Run:

python ml/eda.py
3. ML Recovery Prediction

The first important part of RecoverX is the ML model.

The model predicts:

"How likely is this failed transaction to be recovered?"

I tested:

Logistic Regression
Random Forest

The final model achieved around 75% ROC-AUC on the test data.

The model gives an output like:

{
    "recovery_probability": 73.81,
    "recovery_level": "HIGH"
}

The probability from the ML model is then passed to the AI agents.

To train the model:

python ml/train_model.py

4. Diagnosis Agent

After the ML model predicts the recovery probability, the Diagnosis Agent looks at the transaction in more detail.

I use Groq and an LLM for this part.

The agent looks at:

Why the payment failed
Payment method
Customer payment history
Customer lifetime value
Customer tenure
Time since failure
Previous recovery attempts
Contact permission
ML recovery probability

For example, if the payment failed because of a bank decline and the customer has a good payment history, the agent may identify that another payment method could work better.

The output is structured JSON like:

{
    "diagnosis": "Payment failed due to a bank decline...",
    "recovery_potential": "HIGH",
    "key_factors": [
        "bank_declined",
        "strong payment history",
        "high customer lifetime value"
    ],
    "recommended_strategy": "ALTERNATIVE_PAYMENT"
}

I also added a fallback so that the system does not break if the LLM returns an invalid response.

5. Decision Agent

The Diagnosis Agent explains the situation, but another agent is responsible for making the final recovery decision.

The Decision Agent receives:

Transaction information
ML prediction
Diagnosis Agent output

It can choose from:

RETRY
ALTERNATIVE_PAYMENT
REMINDER
DISCOUNT
ESCALATE
DO_NOT_CONTACT

For example:

{
    "action": "ALTERNATIVE_PAYMENT",
    "confidence": 0.92,
    "reason": "Bank decline suggests using another payment method."
}

I also added validation and fallback logic here so the whole workflow does not depend completely on a successful LLM response.

6. Guardrail Engine

This is one of the parts I wanted to keep deterministic.

I did not want an LLM to directly execute a payment recovery action without checking it first.

The Guardrail Engine checks things such as:

Contact permission

If the customer has not allowed contact:

DO_NOT_CONTACT
Recovery attempts

If the transaction has already been retried too many times:

ESCALATE
Action validation

Only supported actions can reach the executor.

So the basic idea is:

AI Decision
     |
     v
Guardrails
     |
     v
Safe Action

7. Action Executor

Once the Guardrail Engine approves the action, the Action Executor performs the simulated recovery action.

For example:

RETRY
Payment retry initiated
ALTERNATIVE_PAYMENT
Alternative payment methods generated
REMINDER
Payment reminder generated
ESCALATE
Transaction sent for human review

The executor also records the action and timestamp.

Recovery Simulation

After building the individual components, I created a simulation to see how RecoverX would perform across failed and abandoned transactions.

Run:

python simulation/recovery_simulator.py

The simulation calculates:

Total revenue at risk
Recovered revenue
Recovery rate
Recovered transactions
Action distribution
Recovery by failure reason
Recovery by payment method
Baseline Comparison

I also wanted to compare RecoverX with a simpler strategy.

So I created a baseline comparison between:

No Recovery
Naive Recovery
RecoverX

Run:

python simulation/baseline_comparison.py

In my latest simulation:

Revenue at Risk:       ₹2.89 Cr
RecoverX Recovered:    ₹97.90 Lakh
Recovery Rate:         33.91%
Transaction Recovery:  35.05%

Compared with the naive recovery strategy:

Naive Recovery:        ₹67.69 Lakh
RecoverX:              ₹97.90 Lakh


Revenue Uplift:        ₹30.20 Lakh
Uplift Percentage:     44.62%

These are results from my synthetic simulation, so they should not be considered real-world Razorpay performance.

Dashboard

I also built a Streamlit dashboard to make the project easier to understand and demonstrate.

The dashboard shows:

Revenue at risk
Recovered revenue
Recovery rate
Recovery by failure reason
Recovery by payment method
Recommended recovery actions
Baseline vs RecoverX
Live transaction analysis

Run:

streamlit run dashboard/app.py

The most important part of the dashboard is the Live RecoverX Agent.

I can select a failed transaction and run the complete workflow:

Transaction
     ↓
ML Prediction
     ↓
Diagnosis Agent
     ↓
Decision Agent
     ↓
Guardrail Engine
     ↓
Action Executor
     ↓
Final Action
Testing

I also added automated tests using Pytest.

Currently I have 8 tests covering:

ML prediction
Guardrails
Action Executor

Run:

pytest -v

Current result:

8 passed

Some of the things tested include:

Retry execution
Reminder execution
Alternative payment execution
Contact permission
Too many recovery attempts
ML probability output
ML recovery level
Project Structure : 
RazorPay-Buildathon/
│
├── agents/
│   ├── __init__.py
│   ├── diagnosis_agent.py
│   ├── decision_agent.py
│   ├── guardrails.py
│   └── action_executor.py
│
├── analytics/
│   └── metrics.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── generate_dataset.py
│   ├── recovery_by_failure_reason.png
│   └── recovery_by_payment_method.png
│
├── ml/
│   ├── eda.py
│   ├── predict.py
│   ├── train_model.py
│   └── recovery_model.pkl
│
├── simulation/
│   ├── baseline_comparison.py
│   └── recovery_simulator.py
│
├── tests/
│   ├── test_executor.py
│   ├── test_guardrails.py
│   └── test_ml.py
│
├── main.py
├── pytest.ini
├── requirements.txt
└── .gitignore
Technologies I Used
Machine Learning
Python
Scikit-learn
Logistic Regression
Random Forest
Joblib
Generative AI
Groq API
GPT-OSS-120B
Data
Pandas
NumPy
Faker
Dashboard
Streamlit
Plotly
Testing
Pytest