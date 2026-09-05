from flask import Flask,request,render_template_string
import pandas as pd
from sklearn.preprocessing import StandardScaler
from models.anomaly_model import train_model,predict
from preprocessing.preprocess import FEATURES

app=Flask(__name__)
df=pd.read_csv("data/transactions.csv")
scaler=StandardScaler()
X=scaler.fit_transform(df[FEATURES])
model=train_model(X)

HTML="""<!doctype html><html><body style="font-family:Arial;max-width:650px;margin:40px auto">
<h2>AI Transaction Anomaly Detection</h2>
<form method="post">
Amount: <input name="amount" type="number" step=".01" required><br><br>
Frequency: <input name="frequency" type="number" required><br><br>
Hour (0-23): <input name="hour" type="number" min="0" max="23" required><br><br>
Account age (months): <input name="age" type="number" required><br><br>
Location change (0/1): <input name="location" type="number" min="0" max="1" required><br><br>
Average amount: <input name="average" type="number" step=".01" required><br><br>
<button>Analyze</button></form>
{% if result %}<h3>{{result}}</h3><p>Anomaly score: {{score}}</p>{% endif %}
</body></html>"""

@app.route("/",methods=["GET","POST"])
def home():
    result=score=None
    if request.method=="POST":
        vals=[[float(request.form["amount"]),float(request.form["frequency"]),
        float(request.form["hour"]),float(request.form["age"]),
        float(request.form["location"]),float(request.form["average"])]]
        p,s=predict(model,scaler.transform(vals))
        result="⚠️ Anomalous transaction detected" if p[0] else "✅ Transaction appears normal"
        score=round(float(s[0]),4)
    return render_template_string(HTML,result=result,score=score)

if __name__=="__main__":
    app.run(debug=True)
