# AI-Based Anomaly Detection

## Research Title
**AI-Based Anomaly Detection: An Experimental Study Using Machine Learning**

## Objective
This project studies whether an unsupervised machine-learning algorithm can identify unusual transaction patterns without being explicitly trained with anomaly labels.

## Research Question
Does adding multiple behavioral features improve anomaly detection compared with using transaction amount alone?

## Methodology
1. Generate synthetic transaction data.
2. Standardize numerical features.
3. Train an Isolation Forest model.
4. Predict normal and anomalous transactions.
5. Evaluate predictions using accuracy, precision, recall, F1-score and false-positive rate.
6. Compare an amount-only experiment with a multi-feature experiment.
7. Visualize detected anomalies.

## Dataset
The dataset contains 1,000 synthetic transactions:
- 950 normal records
- 50 anomalous records

Features:
- transaction_amount
- transaction_frequency
- transaction_hour
- account_age_months
- location_change
- average_transaction_amount

`is_anomaly` is an evaluation label. Isolation Forest itself is trained without using this label.

**Important:** This is synthetic educational data, not real banking/customer data and not a production fraud-detection system.

## Architecture

Transaction Data
       ↓
Data Preprocessing
       ↓
Feature Scaling
       ↓
Isolation Forest
       ↓
Anomaly Score
       ↓
Normal / Anomalous
       ↓
Evaluation & Visualization

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run Experiment

```bash
python experiments/experiment.py
```

This creates:
- `results/metrics.csv`
- `results/anomaly_plot.png`

## Run Visualization

```bash
python visualization/plots.py
```

## Run Web Application

```bash
python app.py
```

Open:

`http://127.0.0.1:5000`

## Research Interpretation

The project compares:
- **Amount-only model:** uses transaction amount as the only feature.
- **Multi-feature model:** uses transaction amount plus behavioral features.

The experiment is intended to demonstrate how feature selection can affect anomaly detection performance. Results should be reported from the actual execution rather than assumed in advance.

## Limitations
- Synthetic dataset
- Small dataset
- Artificial anomaly patterns
- No real-world banking context
- No production security or fraud investigation workflow

## Future Work
- Test on public fraud datasets.
- Compare Isolation Forest with One-Class SVM and Local Outlier Factor.
- Add ROC/PR analysis where appropriate.
- Study threshold and contamination sensitivity.
- Add explainable anomaly scoring.
