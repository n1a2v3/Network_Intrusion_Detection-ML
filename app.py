import os
import pandas as pd
from flask import Flask, render_template, request
from prediction_utils import load_all, predict_record

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    results = None
    record = None
    metadata, models = load_all()

    if request.method == "POST":
        try:
            if "traffic_file" in request.files and request.files["traffic_file"].filename:
                uploaded = request.files["traffic_file"]
                df = pd.read_csv(uploaded)
                df.columns = [str(c).strip().replace("|", "") for c in df.columns]
                df = df.drop(columns=[c for c in ["label", "attack_type"] if c in df.columns], errors="ignore")
                if df.empty:
                    raise ValueError("The uploaded CSV contains no traffic records.")
                record = df.iloc[0].to_dict()
                results = predict_record(record, metadata, models)
            else:
                import json
                raw = request.form.get("record_json", "").strip()
                record = json.loads(raw)
                if not isinstance(record, dict):
                    raise ValueError("Enter one JSON object representing a single traffic record.")
                results = predict_record(record, metadata, models)
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html", results=results, error=error, record=record,
        feature_columns=metadata["feature_columns"]
    )

@app.route("/health")
def health():
    return {"status": "running", "service": "Network Intrusion Detection"}

if __name__ == "__main__":
    # Local coursework demo only; do not expose the development server publicly.
    app.run(host="127.0.0.1", port=5000, debug=False)
