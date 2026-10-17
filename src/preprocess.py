from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "customer_churn.csv"
)

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop customer identifier if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    # Remove rows with missing values
    df = df.dropna()

    return df

def encode_features(df: pd.DataFrame):
    df = df.copy()

    # Convert target variable to binary
    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Convert categorical features into numeric columns
    X = pd.get_dummies(X, drop_first=True)

    return X, y

def split_and_scale(
    X, y, test_size=0.2, random_state=42
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler
    )

def load_and_preprocess_data():
    # Load and clean data
    df = load_data(str(DATA_PATH))
    df = clean_data(df)

    # Encode features and target
    X, y = encode_features(df)

    # Split and scale data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler
    ) = split_and_scale(X, y)

    print("Training shape:", X_train_scaled.shape)
    print("Testing shape:", X_test_scaled.shape)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train.to_numpy(),
        y_test.to_numpy()
    )

if __name__ == "__main__":
    load_and_preprocess_data()
