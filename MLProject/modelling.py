import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Path dataset 
TRAIN_PATH  = "developer_burnout_preprocessing/train.csv"
TEST_PATH   = "developer_burnout_preprocessing/test.csv"
TARGET      = "burnout_level"

# Nama experiment MLflow
EXPERIMENT  = "developer-burnout-modelling"

# Folder lokal untuk simpan artefak tambahan
ARTIFACT_DIR = "artifacts"


def load_data():
    """Load dataset train & test"""
    train   = pd.read_csv(TRAIN_PATH)
    test    = pd.read_csv(TEST_PATH)
    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]
    X_test  = test.drop(columns=[TARGET])
    y_test  = test[TARGET]
    return X_train, X_test, y_train, y_test


def evaluate(y_true, y_pred) -> dict:
    """Hitung metrik evaluasi"""
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1_macro":  float(f1_score(y_true, y_pred, average="macro")),
        "precision": float(precision_score(y_true, y_pred, average="macro")),
        "recall":    float(recall_score(y_true, y_pred, average="macro")),
    }


def run():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Set experiment MLflow
    mlflow.set_experiment(EXPERIMENT)
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

    # Training model (tanpa mlflow.start_run, karena run sudah otomatis dibuat oleh mlflow run MLProject)
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Prediksi & evaluasi
    y_pred  = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    mlflow.log_metrics(metrics)

    # Log model secara eksplisit ke MLflow
    mlflow.sklearn.log_model(model, "model")

    # Simpan model ke folder artifacts lokal (opsional)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(ARTIFACT_DIR, "model.joblib"))

    # Print hasil evaluasi
    print("=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")


if __name__ == "__main__":
    run()
