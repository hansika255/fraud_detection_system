import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Generate synthetic dataset
np.random.seed(42)
n = 50000

data = pd.DataFrame({
    "transaction_amount": np.random.exponential(scale=2000, size=n),
    "transaction_time_hour": np.random.randint(0, 24, n),
    "transaction_frequency_24h": np.random.poisson(2, n),
    "avg_transaction_amount_7d": np.random.exponential(scale=1500, size=n),
    "location_mismatch": np.random.randint(0, 2, n),
    "device_mismatch": np.random.randint(0, 2, n),
    "international_transaction": np.random.randint(0, 2, n),
    "days_since_account_open": np.random.randint(1, 2000, n),
    "failed_login_attempts_24h": np.random.poisson(1, n)
})

# Fraud logic
fraud_prob = (
    (data["transaction_amount"] > 5000) * 0.3 +
    (data["location_mismatch"] == 1) * 0.2 +
    (data["device_mismatch"] == 1) * 0.2 +
    (data["international_transaction"] == 1) * 0.2 +
    (data["failed_login_attempts_24h"] > 3) * 0.2 +
    (data["transaction_frequency_24h"] > 5) * 0.2 +
    (data["days_since_account_open"] < 30) * 0.2
)

data["fraud_label"] = (fraud_prob > 0.5).astype(int)

X = data.drop("fraud_label", axis=1)
y = data["fraud_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(eval_metric="logloss")
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

pickle.dump(model, open("fraud_model.pkl", "wb"))

print("Model trained and saved.")
