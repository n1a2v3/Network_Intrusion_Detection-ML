import json
import os
import joblib
import pandas as pd

MODEL_DIR = "models"

def load_all():
    metadata_path = os.path.join(MODEL_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Models not found. Run: python train_models.py")
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    models = {}
    for name in metadata["models"]:
        filename = name.lower().replace(" ", "_") + ".joblib"
        models[name] = joblib.load(os.path.join(MODEL_DIR, filename))
    return metadata, models

def predict_record(record, metadata, models):
    features = metadata["feature_columns"]
    row = {column: record.get(column, None) for column in features}
    X = pd.DataFrame([row], columns=features)
    results = {}
    for name, pipeline in models.items():
        if name == "Linear Regression":
            score = float(pipeline.predict(X)[0])
            prediction = int(score >= 0.5)
            results[name] = {
                "prediction": "ATTACK / ANOMALOUS" if prediction else "NORMAL",
                "anomaly_score": round(score, 4)
            }
        else:
            prediction = int(pipeline.predict(X)[0])
            results[name] = {
                "prediction": "ATTACK / ANOMALOUS" if prediction else "NORMAL"
            }
    return results
