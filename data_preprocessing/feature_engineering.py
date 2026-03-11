"""
Feature Engineering Examples
================================
This module demonstrates common feature engineering techniques
used to improve model performance.
"""

import numpy as np
import pandas as pd


def polynomial_features(X, degree=2):
    """Generate polynomial features up to a given degree.

    For a single feature x, generates [x, x^2, ..., x^degree].

    Args:
        X: Input array, shape (n_samples, n_features)
        degree: Maximum polynomial degree

    Returns:
        Array with original and polynomial features
    """
    features = [X]
    for d in range(2, degree + 1):
        features.append(X ** d)
    return np.hstack(features)


def interaction_features(X):
    """Generate pairwise interaction features.

    For features [x1, x2, x3], generates [x1*x2, x1*x3, x2*x3].

    Args:
        X: Input array, shape (n_samples, n_features)

    Returns:
        Array with pairwise interaction features appended
    """
    n_features = X.shape[1]
    interactions = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
    return np.hstack([X] + interactions)


def bin_feature(x, n_bins=5):
    """Discretize a continuous feature into bins.

    Args:
        x: 1D array of continuous values
        n_bins: Number of bins to create

    Returns:
        Array of bin indices (0 to n_bins-1)
    """
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    return np.digitize(x, bins[1:-1])


def log_transform(X, epsilon=1e-8):
    """Apply log transformation to handle skewed distributions.

    Args:
        X: Input array (all values must be non-negative)
        epsilon: Small value to avoid log(0)

    Returns:
        Log-transformed array
    """
    return np.log(X + epsilon)


def date_features(df, date_column):
    """Extract useful features from a datetime column.

    Extracts year, month, day, day-of-week, and hour (if present).

    Args:
        df: Pandas DataFrame with a datetime column
        date_column: Name of the datetime column

    Returns:
        DataFrame with extracted date features added
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df[f"{date_column}_year"] = df[date_column].dt.year
    df[f"{date_column}_month"] = df[date_column].dt.month
    df[f"{date_column}_day"] = df[date_column].dt.day
    df[f"{date_column}_dayofweek"] = df[date_column].dt.dayofweek
    df[f"{date_column}_hour"] = df[date_column].dt.hour
    df.drop(columns=[date_column], inplace=True)
    return df


def main():
    np.random.seed(42)
    X = np.random.randn(10, 2)

    print("Original features shape:", X.shape)

    X_poly = polynomial_features(X, degree=3)
    print("Polynomial features shape:", X_poly.shape)

    X_interact = interaction_features(X)
    print("Interaction features shape:", X_interact.shape)

    x = np.array([1.5, 2.3, 0.8, 4.1, 3.7, 1.2, 5.0, 0.5, 2.9, 3.3])
    binned = bin_feature(x, n_bins=5)
    print("\nOriginal values:", x)
    print("Binned values:  ", binned)

    X_positive = np.abs(X) + 1
    X_log = log_transform(X_positive)
    print("\nLog transform applied. Original range:", (X_positive.min(), X_positive.max()))
    print("Log-transformed range:", (X_log.min(), X_log.max()))

    df = pd.DataFrame({
        "timestamp": ["2023-01-15 08:30:00", "2023-06-20 14:45:00", "2023-12-31 23:59:00"],
        "value": [10, 20, 30],
    })
    df_features = date_features(df, "timestamp")
    print("\nDate features extracted:")
    print(df_features)


if __name__ == "__main__":
    main()
