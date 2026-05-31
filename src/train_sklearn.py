import os
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from preprocess import load_data, clean_data, encode_features, split_and_scale


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Some models support probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)

    return metrics


def main():
    csv_path = os.path.join("data", "customer_churn.csv")

    df = load_data(csv_path)
    df = clean_data(df)
    X, y = encode_features(df)

    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = split_and_scale(X, y)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}

    # Logistic regression uses scaled features
    lr = models["logistic_regression"]
    lr.fit(X_train_scaled, y_train)
    results["logistic_regression"] = evaluate_model("logistic_regression", lr, X_test_scaled, y_test)

    # Random forest can use unscaled features
    rf = models["random_forest"]
    rf.fit(X_train, y_train)
    results["random_forest"] = evaluate_model("random_forest", rf, X_test, y_test)

    print(json.dumps(results, indent=2))

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/sklearn_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

