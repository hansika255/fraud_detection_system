from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and infer feature list
base_dir = os.path.dirname(os.path.abspath(__file__))
model = None
feature_names = None
model_candidates = ['fraud_model.pkl', 'fraud_detection_model.pkl']
for candidate in model_candidates:
    path = os.path.join(base_dir, candidate)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        break

if model is not None:
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)

# Fallback feature list (matches front-end sample data)
if not feature_names:
    feature_names = [
        'transaction_amount', 'transaction_time_hour', 'transaction_frequency_24h',
        'avg_transaction_amount_7d', 'location_mismatch', 'device_mismatch',
        'international_transaction', 'days_since_account_open', 'failed_login_attempts_24h'
    ]


@app.route("/")
def home():
    return render_template("index.html", features=feature_names)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["transaction_amount"]),
            float(request.form["transaction_time_hour"]),
            float(request.form["transaction_frequency_24h"]),
            float(request.form["avg_transaction_amount_7d"]),
            float(request.form["location_mismatch"]),
            float(request.form["device_mismatch"]),
            float(request.form["international_transaction"]),
            float(request.form["days_since_account_open"]),
            float(request.form["failed_login_attempts_24h"])
        ]

        final_features = np.array([features])
        probability = model.predict_proba(final_features)[0][1]
        probability_percent = round(probability * 100, 2)

        if probability < 0.3:
            risk = "Low Risk"
            color = "green"
        elif probability < 0.7:
            risk = "Medium Risk"
            color = "orange"
        else:
            risk = "High Risk"
            color = "red"

        return render_template(
            "index.html",
            probability=probability_percent,
            risk=risk,
            color=color,
            features=feature_names
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
