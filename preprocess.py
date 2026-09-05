import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES=["transaction_amount","transaction_frequency","transaction_hour",
"account_age_months","location_change","average_transaction_amount"]

def load_data(path="data/transactions.csv"):
    return pd.read_csv(path)

def prepare_features(df, features=FEATURES):
    scaler=StandardScaler()
    X=scaler.fit_transform(df[features])
    return X, scaler
