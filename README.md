# FinGuard AI - Fraud Detection System

A machine learning-based fraud detection system built with Flask and scikit-learn.

## Project Structure

```
fraud_detection_system/
├── app.py                 # Flask web application
├── model.py              # Model training script
├── test_project.py       # Project audit and testing script
├── fraud_model.pkl       # Trained Random Forest model (binary)
├── feature_names.pkl     # Feature names used by the model
├── dataset.csv           # Data loading helper
├── templates/
│   └── index.html        # Web UI template
├── static/
│   └── style.css         # Custom styling
└── README.md             # This file
```

## Features

- **29 Transaction Features**: V1-V28 (PCA-transformed features) + Amount
- **Random Forest Classifier**: 50 trees, depth=10
- **Model Performance**: 98.5% accuracy on test data
- **Real-time Predictions**: Web interface for immediate fraud risk analysis
- **Risk Classification**: Low Risk (< 30%), Medium Risk (30-70%), High Risk (> 70%)

## Installation

### Prerequisites
- Python 3.7+
- Virtual environment (recommended)

### Setup

1. Create and activate virtual environment:
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

2. Install dependencies:
```bash
pip install flask pandas scikit-learn numpy kagglehub
```

3. Verify the installation:
```bash
python test_project.py
```

Expected output:
```
============================================================
✓ PROJECT STATUS: ALL SYSTEMS GO
✓ Ready to launch Flask server
============================================================
```

## Usage

### Train the Model

```bash
python model.py
```

This will:
- Download the credit card fraud detection dataset from Kaggle
- Train a Random Forest model with 29 features
- Save the model as `fraud_model.pkl`
- Save feature names as `feature_names.pkl`

### Run the Web Application

```bash
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/predict` | POST | Submit transaction data for fraud prediction |

## Model Details

- **Algorithm**: Random Forest Classifier
- **Number of Trees**: 50
- **Max Depth**: 10
- **Input Features**: 29
- **Output Classes**: [0 = Legitimate, 1 = Fraud]
- **Training Data**: 568,630 transactions
- **Train/Test Split**: 80/20
- **Accuracy**: 98.50%
- **Precision (Fraud)**: 100%
- **Recall (Fraud)**: 97%

## Feature Descriptions

The model uses 28 principal component-transformed features (V1-V28) plus the transaction amount:

- **V1-V28**: PCA-transformed credit card transaction features
- **Amount**: Transaction amount in USD

Note: The 'id' column and 'Class' target column are automatically excluded from the model input.

## Performance Metrics (Test Set)

```
             precision    recall  f1-score   support
       0       0.97      1.00      0.99     56750
       1       1.00      0.97      0.98     56976
accuracy                           0.99    113726
```

## Troubleshooting

### Missing Dependencies
If you get import errors, install the required packages:
```bash
pip install -r requirements.txt
```

### Model File Not Found
The model trains automatically if not present:
```bash
python model.py
```

### Port Already in Use
Change the port in `app.py`:
```python
app.run(debug=True, port=5001)
```

### Kaggle Authentication Error
The dataset downloads automatically if you have internet access. For offline mode, provide a local `dataset.csv` file with the proper format.

## Variables in Input Form

When using the web interface, enter 29 numerical values for:
- V1 through V28 (transaction features, can be positive or negative)
- Amount (transaction amount, typically positive)

## Security Notes

- This is a demonstration system suitable for development and testing
- For production use, implement:
  - HTTPS/SSL encryption
  - Rate limiting
  - Authentication
  - Input validation and sanitization
  - Logging and monitoring

## Dataset Source

[Credit Card Fraud Detection Dataset (2023)](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)

## License

This project is provided as-is for educational and demonstration purposes.

## Support

For issues or questions:
1. Run `python test_project.py` to verify project integrity
2. Check that all dependencies are installed
3. Ensure the model files exist (`fraud_model.pkl`, `feature_names.pkl`)
4. Verify Flask is running without errors

---

**Last Updated**: February 16, 2026
**Status**: ✓ All Systems Operational
