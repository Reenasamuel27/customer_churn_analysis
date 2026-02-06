from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("../models/churn_model.pkl")
features = joblib.load("../models/feature_columns.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data]).reindex(columns=features, fill_value=0)
    prob = model.predict_proba(df)[0][1]
    return {"churn_risk": float(prob)}
