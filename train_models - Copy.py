import json
import os
import sys
import warnings
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def read_csv_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find {path}. Put your dataset in the data folder "
            "and name it train.csv or test.csv."
        )
    # The supplied coursework data should be saved as a normal CSV file.
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip().replace("|", "") for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]
    return df

def make_binary_target(df):
    if "label" in df.columns:
        y = df["label"]
        if pd.api.types.is_numeric_dtype(y):
            values = set(pd.Series(y.dropna().unique()).tolist())
            if values.issubset({0, 1}):
                return y.astype(int)
        text = y.astype(str).str.strip().str.lower()
        return (text != "normal").astype(int)
    if "attack_type" in df.columns:
        return (df["attack_type"].astype(str).str.strip().str.lower() != "normal").astype(int)
    raise ValueError("Dataset must contain a 'label' or 'attack_type' target column.")

def prepare_features(df, feature_columns=None):
    # Neither label nor attack_type is allowed to be a model input.
    X = df.drop(columns=[c for c in ["label", "attack_type"] if c in df.columns])
    if feature_columns is not None:
        X = X.reindex(columns=feature_columns)
    # Convert numeric-looking columns to numbers where possible.
    for col in X.columns:
        if X[col].dtype == "object":
            converted = pd.to_numeric(X[col], errors="coerce")
            if converted.notna().sum() >= max(1, int(0.8 * X[col].notna().sum())):
                X[col] = converted
    return X

def build_preprocessor(X, scale_numeric):
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipe, categorical_cols))
    return ColumnTransformer(transformers, remainder="drop")

def binary_metrics(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

def main():
    print("=" * 65)
    print("       INTELLIGENT NETWORK INTRUSION DETECTION SYSTEM")
    print("=" * 65)

    train_df = read_csv_file(os.path.join(DATA_DIR, "train.csv"))
    test_df = read_csv_file(os.path.join(DATA_DIR, "test.csv"))

    y_train = make_binary_target(train_df)
    y_test = make_binary_target(test_df)
    X_train = prepare_features(train_df)
    feature_columns = X_train.columns.tolist()
    X_test = prepare_features(test_df, feature_columns)

    if y_train.nunique() < 2:
        raise ValueError("Training data must contain both normal and attack records.")
    if y_test.nunique() < 2:
        print("WARNING: Test data contains only one target class; some metrics may be less informative.")

    print(f"\nTraining records: {len(X_train):,}")
    print(f"Testing records : {len(X_test):,}")
    print(f"Input features  : {len(feature_columns)}")
    print("\nTraining label distribution:")
    print(y_train.value_counts().rename(index={0: "Normal", 1: "Attack"}))

    model_definitions = {
        "Linear Regression": (
            Pipeline([
                ("preprocessor", build_preprocessor(X_train, True)),
                ("model", LinearRegression())
            ]),
            "regression"
        ),
        "Random Forest": (
            Pipeline([
                ("preprocessor", build_preprocessor(X_train, False)),
                ("model", RandomForestClassifier(
                    n_estimators=150, random_state=42,
                    class_weight="balanced", n_jobs=-1
                ))
            ]),
            "classification"
        ),
        "Neural Network": (
            Pipeline([
                ("preprocessor", build_preprocessor(X_train, True)),
                ("model", MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    max_iter=150,
                    early_stopping=True,
                    random_state=42
                ))
            ]),
            "classification"
        )
    }

    all_results = {}
    for name, (pipeline, model_type) in model_definitions.items():
        print("\n" + "-" * 65)
        print(f"TRAINING: {name}")
        pipeline.fit(X_train, y_train)

        if model_type == "regression":
            scores = pipeline.predict(X_test)
            predictions = (scores >= 0.5).astype(int)
        else:
            predictions = pipeline.predict(X_test)

        metrics = binary_metrics(y_test, predictions)
        all_results[name] = metrics
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1-score : {metrics['f1_score']:.4f}")
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(metrics["confusion_matrix"])
        print("\nClassification report:")
        print(classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=["Normal", "Attack"], zero_division=0
        ))

        safe_name = name.lower().replace(" ", "_")
        joblib.dump(pipeline, os.path.join(MODEL_DIR, safe_name + ".joblib"))

    metadata = {
        "feature_columns": feature_columns,
        "models": list(model_definitions.keys()),
        "results": all_results,
        "target_definition": "0 = Normal, 1 = Attack/Anomalous",
        "linear_regression_threshold": 0.5
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print("Saved trained models in the 'models' folder.")
    print("Next: run python app.py to open the web dashboard.")
    print("=" * 65)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
