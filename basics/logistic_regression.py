"""
Logistic Regression from Scratch
==================================
This module demonstrates how to implement logistic regression
for binary classification using only NumPy.
"""

import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    """Logistic regression using gradient descent."""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """Apply the sigmoid activation function.

        Args:
            z: Input value or array

        Returns:
            Sigmoid of z, in range (0, 1)
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """Train the model using gradient descent.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Binary target values (0 or 1), shape (n_samples,)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        """Predict class probabilities for given input features.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Predicted probabilities, shape (n_samples,)
        """
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """Predict class labels for given input features.

        Args:
            X: Input features, shape (n_samples, n_features)
            threshold: Decision boundary (default 0.5)

        Returns:
            Predicted class labels (0 or 1), shape (n_samples,)
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, y_true, y_pred):
        """Calculate classification accuracy.

        Args:
            y_true: True class labels
            y_pred: Predicted class labels

        Returns:
            Accuracy score (float between 0 and 1)
        """
        return np.mean(y_true == y_pred)


def main():
    np.random.seed(42)
    n_samples = 200
    X_class0 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    X_class1 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

    shuffle_idx = np.random.permutation(n_samples)
    X, y = X[shuffle_idx], y[shuffle_idx]

    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = model.accuracy(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color="blue", label="Class 0")
    plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color="red", label="Class 1")
    plt.title("Logistic Regression Classification")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("logistic_regression_plot.png")
    print("Plot saved to logistic_regression_plot.png")


if __name__ == "__main__":
    main()
