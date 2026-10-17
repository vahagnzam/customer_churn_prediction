import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.preprocess import (
    load_data,
    clean_data,
    encode_features,
    split_and_scale,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "customer_churn.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))

    return metrics

def main():
    df = load_data(str(CSV_PATH))
    df = clean_data(df)

    X, y = encode_features(df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler,
    ) = split_and_scale(X, y)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        ),
    }

    results = {}

    # Logistic regression uses scaled features
    lr = models["logistic_regression"]
    lr.fit(X_train_scaled, y_train)

    results["logistic_regression"] = evaluate_model(
        "logistic_regression",
        lr,
        X_test_scaled,
        y_test,
    )

    # Random forest uses unscaled features
    rf = models["random_forest"]
    rf.fit(X_train, y_train)

    results["random_forest"] = evaluate_model(
        "random_forest",
        rf,
        X_test,
        y_test,
    )

    print(json.dumps(results, indent=2))

    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_DIR / "sklearn_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results to outputs/sklearn_results.json")

if __name__ == "__main__":
    main()
