"""
Data Preprocessing Techniques
================================
This module demonstrates common data preprocessing techniques
used in machine learning pipelines.
"""

import numpy as np
import pandas as pd


def normalize_min_max(X):
    """Normalize features to the range [0, 1].

    Args:
        X: Input array, shape (n_samples, n_features)

    Returns:
        Normalized array with values in [0, 1]
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-8)


def standardize(X):
    """Standardize features to have zero mean and unit variance.

    Args:
        X: Input array, shape (n_samples, n_features)

    Returns:
        Standardized array with mean=0 and std=1
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + 1e-8)


def encode_labels(y):
    """Encode categorical labels to integer indices.

    Args:
        y: Array of categorical labels

    Returns:
        Tuple of (encoded_labels, label_mapping) where label_mapping maps
        original labels to integer indices
    """
    unique_labels = np.unique(y)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = np.array([label_mapping[label] for label in y])
    return encoded, label_mapping


def one_hot_encode(y, n_classes=None):
    """One-hot encode integer class labels.

    Args:
        y: Integer class labels, shape (n_samples,)
        n_classes: Number of classes (inferred from y if None)

    Returns:
        One-hot encoded array, shape (n_samples, n_classes)
    """
    if n_classes is None:
        n_classes = int(y.max()) + 1
    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y.astype(int)] = 1
    return one_hot


def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split arrays into random train and test subsets.

    Args:
        X: Input features, shape (n_samples, n_features)
        y: Target values, shape (n_samples,)
        test_size: Fraction of samples for the test set (default 0.2)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    n_test = int(n_samples * test_size)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def handle_missing_values(df, strategy="mean"):
    """Fill missing values in a DataFrame.

    Args:
        df: Pandas DataFrame with potential NaN values
        strategy: Imputation strategy - "mean", "median", or "mode"

    Returns:
        DataFrame with missing values filled
    """
    df_filled = df.copy()
    for column in df_filled.select_dtypes(include=[np.number]).columns:
        if strategy == "mean":
            df_filled[column].fillna(df_filled[column].mean(), inplace=True)
        elif strategy == "median":
            df_filled[column].fillna(df_filled[column].median(), inplace=True)
        elif strategy == "mode":
            df_filled[column].fillna(df_filled[column].mode()[0], inplace=True)
    return df_filled


def main():
    np.random.seed(42)
    X = np.random.randn(100, 3) * 10 + 5
    X[::10, 0] = np.nan

    df = pd.DataFrame(X, columns=["feature_1", "feature_2", "feature_3"])
    print("Original data (first 5 rows):")
    print(df.head())
    print(f"\nMissing values:\n{df.isnull().sum()}")

    df_clean = handle_missing_values(df, strategy="mean")
    print(f"\nAfter filling missing values:\n{df_clean.isnull().sum()}")

    X_clean = df_clean.values
    X_normalized = normalize_min_max(X_clean)
    print(f"\nNormalized range: [{X_normalized.min():.4f}, {X_normalized.max():.4f}]")

    X_standardized = standardize(X_clean)
    print(f"Standardized mean: {X_standardized.mean():.4f}, std: {X_standardized.std():.4f}")

    labels = np.array(["cat", "dog", "bird", "cat", "dog"])
    encoded, mapping = encode_labels(labels)
    print(f"\nLabel encoding: {mapping}")
    print(f"Encoded labels: {encoded}")

    one_hot = one_hot_encode(encoded)
    print(f"One-hot encoded:\n{one_hot}")


if __name__ == "__main__":
    main()
