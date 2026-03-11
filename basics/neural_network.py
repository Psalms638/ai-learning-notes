"""
Neural Network from Scratch
=============================
This module demonstrates how to implement a simple feedforward
neural network with one hidden layer using only NumPy.
"""

import numpy as np


class NeuralNetwork:
    """Simple feedforward neural network with one hidden layer."""

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate

        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        s = self._sigmoid(z)
        return s * (1 - s)

    def _forward(self, X):
        """Perform a forward pass through the network.

        Args:
            X: Input features, shape (n_samples, input_size)

        Returns:
            Tuple of (output, cache) where cache stores intermediate values
        """
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._sigmoid(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self._sigmoid(Z2)

        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def _backward(self, X, y, cache):
        """Perform a backward pass and compute gradients.

        Args:
            X: Input features, shape (n_samples, input_size)
            y: True labels, shape (n_samples, output_size)
            cache: Dictionary with intermediate values from forward pass

        Returns:
            Dictionary with gradients for all parameters
        """
        n_samples = X.shape[0]
        A1, A2 = cache["A1"], cache["A2"]
        Z1 = cache["Z1"]

        dA2 = A2 - y
        dW2 = (1 / n_samples) * np.dot(A1.T, dA2)
        db2 = (1 / n_samples) * np.sum(dA2, axis=0, keepdims=True)

        dA1 = np.dot(dA2, self.W2.T)
        dZ1 = dA1 * self._sigmoid_derivative(Z1)
        dW1 = (1 / n_samples) * np.dot(X.T, dZ1)
        db1 = (1 / n_samples) * np.sum(dZ1, axis=0, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def fit(self, X, y, n_iterations=1000):
        """Train the neural network.

        Args:
            X: Training features, shape (n_samples, input_size)
            y: Target values, shape (n_samples, output_size)
            n_iterations: Number of training iterations
        """
        for i in range(n_iterations):
            output, cache = self._forward(X)
            grads = self._backward(X, y, cache)

            self.W1 -= self.learning_rate * grads["dW1"]
            self.b1 -= self.learning_rate * grads["db1"]
            self.W2 -= self.learning_rate * grads["dW2"]
            self.b2 -= self.learning_rate * grads["db2"]

            if i % 100 == 0:
                loss = self._cross_entropy_loss(y, output)
                print(f"Iteration {i}: Loss = {loss:.4f}")

    def predict(self, X):
        """Predict class labels for given input features.

        Args:
            X: Input features, shape (n_samples, input_size)

        Returns:
            Predicted class labels, shape (n_samples,)
        """
        output, _ = self._forward(X)
        return (output >= 0.5).astype(int).flatten()

    def _cross_entropy_loss(self, y_true, y_pred):
        """Calculate cross-entropy loss.

        Args:
            y_true: True labels
            y_pred: Predicted probabilities

        Returns:
            Cross-entropy loss (float)
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def main():
    np.random.seed(42)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    print("Training XOR Neural Network...")
    model = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
    model.fit(X, y, n_iterations=1000)

    predictions = model.predict(X)
    print("\nXOR Predictions:")
    for i, (inputs, pred) in enumerate(zip(X, predictions)):
        print(f"  Input: {inputs} -> Predicted: {pred}, Actual: {int(y[i][0])}")


if __name__ == "__main__":
    main()
