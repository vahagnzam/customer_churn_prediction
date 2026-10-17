import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

def main():
    # Load the scikit-learn model results
    with open(RESULTS_DIR / "sklearn_results.json") as f:
        sklearn_results = json.load(f)

    # Load the PyTorch neural network results
    with open(RESULTS_DIR / "pytorch_metrics.json") as f:
        pytorch_results = json.load(f)

    # Combine the results into one dictionary
    all_results = {
        **sklearn_results,
        **pytorch_results,
    }

    # Convert results into a pandas DataFrame
    df = pd.DataFrame.from_dict(
        all_results,
        orient="index"
    )

    # Give the models readable names
    df = df.rename(
        index={
            "logistic_regression": "Logistic Regression",
            "random_forest": "Random Forest",
            "PyTorch Neural Network": "PyTorch Neural Network",
        }
    )

    df.index.name = "Model"

    # Use a consistent order for evaluation metrics
    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    df = df[metric_columns]

    # Display comparison table
    print("\nModel Performance Comparison:")
    print(df.round(4).to_string())

    # Save results as a CSV
    csv_path = RESULTS_DIR / "model_comparison.csv"
    df.to_csv(csv_path)

    # Create a comparison chart
    ax = df.plot(
        kind="bar",
        figsize=(12, 6),
        rot=0,
        width=0.8,
    )

    ax.set_title("Customer Churn Model Comparison")
    ax.set_xlabel("Machine Learning Model")
    ax.set_ylabel("Performance Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Evaluation Metric")

    plt.tight_layout()

    # Save chart as an image
    chart_path = RESULTS_DIR / "model_comparison.png"
    plt.savefig(chart_path, dpi=150)
    plt.show()

    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved chart to: {chart_path}")

if __name__ == "__main__":
    main()
