import os,pandas as pd
from sklearn.preprocessing import StandardScaler
from preprocessing.preprocess import load_data,FEATURES
from models.anomaly_model import train_model,predict
from evaluation.evaluate import evaluate

os.makedirs("results",exist_ok=True)

def run_experiment(mode):
    df=load_data()
    features=["transaction_amount"] if mode=="amount_only" else FEATURES
    X=StandardScaler().fit_transform(df[features])
    model=train_model(X)
    pred,scores=predict(model,X)
    metrics=evaluate(df["is_anomaly"],pred)
    metrics["experiment"]=mode
    result=df.copy()
    result["predicted_anomaly"]=pred
    result["anomaly_score"]=scores
    return metrics,result

if __name__=="__main__":
    a,ra=run_experiment("amount_only")
    b,rb=run_experiment("all")
    pd.DataFrame([a,b]).to_csv("results/experiment_comparison.csv",index=False)
    rb.to_csv("results/anomaly_predictions.csv",index=False)
    pd.DataFrame([b]).to_csv("results/metrics.csv",index=False)
    print(pd.DataFrame([a,b]).to_string(index=False))
