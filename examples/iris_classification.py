"""
Iris Dataset Classification Example
======================================
This example demonstrates a complete ML pipeline on the classic Iris dataset,
including data loading, preprocessing, model training, and evaluation.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def load_and_explore(data):
    """Print basic information about the dataset."""
    print("=== Iris Dataset ===")
    print(f"Samples: {data.data.shape[0]}, Features: {data.data.shape[1]}")
    print(f"Classes: {list(data.target_names)}")
    print(f"Feature names: {list(data.feature_names)}\n")


def plot_confusion_matrix(cm, class_names, filename="confusion_matrix.png"):
    """Plot and save a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array
        class_names: List of class name strings
        filename: Output filename for the saved plot
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.tight_layout()
    plt.savefig(filename)
    print(f"Confusion matrix saved to {filename}")


def main():
    iris = load_iris()
    load_and_explore(iris)

    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, iris.target_names)


if __name__ == "__main__":
    main()
