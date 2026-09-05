from sklearn.ensemble import IsolationForest

def train_model(X, contamination=0.05):
    model=IsolationForest(n_estimators=200,contamination=contamination,random_state=42)
    model.fit(X)
    return model

def predict(model,X):
    raw=model.predict(X)
    labels=(raw==-1).astype(int)
    scores=-model.decision_function(X)
    return labels,scores
