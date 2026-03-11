"""
Linear Regression from Scratch
================================
This module demonstrates how to implement simple linear regression
using only NumPy, without relying on scikit-learn.
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    """Simple linear regression using gradient descent."""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """Train the model using gradient descent.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Target values, shape (n_samples,)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """Predict target values for given input features.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Predicted values, shape (n_samples,)
        """
        return np.dot(X, self.weights) + self.bias

    def mean_squared_error(self, y_true, y_pred):
        """Calculate Mean Squared Error.

        Args:
            y_true: True target values
            y_pred: Predicted target values

        Returns:
            Mean Squared Error (float)
        """
        return np.mean((y_true - y_pred) ** 2)


def main():
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X[:, 0] + np.random.randn(100)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = model.mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Learned weights: {model.weights}")
    print(f"Learned bias: {model.bias:.4f}")

    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.plot(X_test, y_pred, color="red", label="Predicted")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression")
    plt.legend()
    plt.tight_layout()
    plt.savefig("linear_regression_plot.png")
    print("Plot saved to linear_regression_plot.png")


if __name__ == "__main__":
    main()
