

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load AllCars.csv and clean required columns."""
    df = pd.read_csv(csv_path)

    # Keep rows that have the required values
    df = df.dropna(subset=["Volume", "Doors", "Style"])

    # Ensure numeric (coerce bad values to NaN, then drop)
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df["Doors"] = pd.to_numeric(df["Doors"], errors="coerce")
    df = df.dropna(subset=["Volume", "Doors"])

    # Clean style strings
    df["Style"] = df["Style"].astype(str).str.strip()

    return df


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42):
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )


def normalize_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def find_best_k(X_train_scaled, y_train, X_test_scaled, y_test, k_min=1, k_max=25):
    rows = []
    best_k = None
    best_acc = -1
    best_model = None

    for k in range(k_min, k_max + 1):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)

        rows.append({"K": k, "Accuracy": acc})

        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_model = model

    accuracy_df = pd.DataFrame(rows)
    return accuracy_df, best_model, best_k, best_acc


def main():
    # 1) Load data
    df = load_and_clean("AllCars.csv")

    # 2) Select numeric features only (ignore Make)
    X = df[["Volume", "Doors"]]
    y = df["Style"]

    # 3) Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4) Normalize
    X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)

    # 5) Find best K and save Accuracy.csv
    accuracy_df, best_model, best_k, best_acc = find_best_k(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    accuracy_df.to_csv("Accuracy.csv", index=False)

    # 6) Predictions + confidence for Testing.csv
    probs = best_model.predict_proba(X_test_scaled)
    confidence = probs.max(axis=1)
    prediction = best_model.predict(X_test_scaled)

    # 7) Save Training.csv 
    train_out = pd.DataFrame(X_train_scaled, columns=["Volume", "Doors"])
    train_out["Style"] = y_train.values
    train_out.to_csv("Training.csv", index=False)

    # 8) Save Testing.csv
    test_out = pd.DataFrame(X_test_scaled, columns=["Volume", "Doors"])
    test_out["Style"] = y_test.values
    test_out["Prediction"] = prediction
    test_out["Confidence"] = confidence
    test_out.to_csv("Testing.csv", index=False)

    print(f"Best K: {best_k}  Best Accuracy: {best_acc}")


if __name__ == "__main__":
    main()
