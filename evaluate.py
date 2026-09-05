from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import pandas as pd

def evaluate(y_true,y_pred):
    tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    return {
        "accuracy":accuracy_score(y_true,y_pred),
        "precision":precision_score(y_true,y_pred,zero_division=0),
        "recall":recall_score(y_true,y_pred,zero_division=0),
        "f1_score":f1_score(y_true,y_pred,zero_division=0),
        "anomaly_detection_rate":float(y_pred.mean()),
        "false_positive_rate":fp/(fp+tn) if fp+tn else 0,
        "true_negatives":tn,"false_positives":fp,
        "false_negatives":fn,"true_positives":tp
    }

def save_metrics(metrics,path="results/metrics.csv"):
    pd.DataFrame([metrics]).to_csv(path,index=False)
