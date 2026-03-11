"""
Housing Prices Prediction Example
=====================================
This example demonstrates a regression pipeline using the California
Housing dataset with feature preprocessing and model evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score


def load_and_explore(data):
    """Print basic information about the dataset."""
    print("=== California Housing Dataset ===")
    print(f"Samples: {data.data.shape[0]}, Features: {data.data.shape[1]}")
    print(f"Feature names: {list(data.feature_names)}")
    print(f"Target: Median house value (in $100,000s)")
    print(f"Target range: [{data.target.min():.2f}, {data.target.max():.2f}]\n")


def evaluate_model(model_name, y_test, y_pred):
    """Print regression evaluation metrics.

    Args:
        model_name: Human-readable model name string
        y_test: True target values
        y_pred: Predicted target values
    """
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name}:")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  R²:    {r2:.4f}\n")
    return rmse, r2


def plot_predictions(y_test, y_pred_lr, y_pred_ridge, filename="housing_predictions.png"):
    """Plot actual vs. predicted values for both models.

    Args:
        y_test: True target values
        y_pred_lr: Predictions from Linear Regression
        y_pred_ridge: Predictions from Ridge Regression
        filename: Output filename for the saved plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, y_pred, title in zip(
        axes,
        [y_pred_lr, y_pred_ridge],
        ["Linear Regression", "Ridge Regression"],
    ):
        ax.scatter(y_test, y_pred, alpha=0.3)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect fit")
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")


def main():
    housing = fetch_california_housing()
    load_and_explore(housing)

    X, y = housing.data, housing.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)

    evaluate_model("Linear Regression", y_test, y_pred_lr)
    evaluate_model("Ridge Regression", y_test, y_pred_ridge)

    plot_predictions(y_test, y_pred_lr, y_pred_ridge)


if __name__ == "__main__":
    main()
