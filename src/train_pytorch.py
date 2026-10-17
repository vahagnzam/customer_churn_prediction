import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.models.pytorch_model import ChurnNet
from src.preprocess import load_and_preprocess_data

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def main():
    # Reproducible initialization
    torch.manual_seed(42)

    # Load the SAME train/test split as scikit-learn
    X_train, X_test, y_train, y_test = (
        load_and_preprocess_data()
    )

    # Convert NumPy arrays into PyTorch tensors
    X_train_tensor = torch.tensor(
        X_train, dtype=torch.float32
    )
    X_test_tensor = torch.tensor(
        X_test, dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train, dtype=torch.float32
    ).reshape(-1, 1)

    # Create network
    model = ChurnNet(input_size=X_train.shape[1])

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )

    # Train
    epochs = 50

    for epoch in range(epochs):
        model.train()

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor)

        # Calculate loss
        loss = criterion(outputs, y_train_tensor)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Loss: {loss.item():.4f}"
            )

    # Evaluate on held-out test data
    model.eval()

    with torch.no_grad():
        logits = model(X_test_tensor)
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.numpy().flatten()

    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_test, predictions, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
    }

    print("\nPyTorch Neural Network Results:")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    output_path = RESULTS_DIR / "pytorch_metrics.json"

    with open(output_path, "w") as file:
        json.dump(
            {"PyTorch Neural Network": metrics},
            file,
            indent=4,
        )

    print(f"\nSaved results to {output_path}")

if __name__ == "__main__":
    main()
