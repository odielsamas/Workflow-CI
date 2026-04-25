import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

TRAIN_PATH  = "developer_burnout_preprocessing/train.csv"
TEST_PATH   = "developer_burnout_preprocessing/test.csv"
TARGET      = "burnout_level"
EXPERIMENT  = "developer-burnout-modelling"
ARTIFACT_DIR = "artifacts"


def load_data():
    train   = pd.read_csv(TRAIN_PATH)
    test    = pd.read_csv(TEST_PATH)
    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]
    X_test  = test.drop(columns=[TARGET])
    y_test  = test[TARGET]
    return X_train, X_test, y_train, y_test


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1_macro":  float(f1_score(y_true, y_pred, average="macro")),
        "precision": float(precision_score(y_true, y_pred, average="macro")),
        "recall":    float(recall_score(y_true, y_pred, average="macro")),
    }


def run():
    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_experiment(EXPERIMENT)
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

    with mlflow.start_run(run_name="RandomForest-autolog"):
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        mlflow.log_metrics(metrics)

        print("=== Evaluation Results ===")
        for k, v in metrics.items():
            print(f"  {k:<12}: {v:.4f}")


if __name__ == "__main__":
    run()
