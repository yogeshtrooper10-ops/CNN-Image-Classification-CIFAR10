# Evaluate the trained CIFAR-10 CNN model.
# This script loads the saved model, evaluates it on test data, saves
# confusion matrix and classification report, and prints results.

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# Project paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(MODEL_DIR, "cifar10_cnn_model.keras")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
CLASSIFICATION_REPORT_PATH = os.path.join(RESULTS_DIR, "classification_report.txt")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# -----------------------------
# Load and evaluate model
# -----------------------------
def main():
    """
    Evaluation pipeline:
    1. Load trained model
    2. Load CIFAR-10 test dataset
    3. Make predictions on test data
    4. Calculate accuracy
    5. Generate confusion matrix
    6. Generate classification report
    7. Save results to files
    """
    print("Loading saved model...")

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at: {MODEL_PATH}")
        print("Please train the model first using train.py.")
        return

    # Load model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully from: {MODEL_PATH}")
    except Exception as exc:
        print(f"Could not load model: {exc}")
        return

    # Load test dataset
    print("Loading CIFAR-10 test dataset...")
    try:
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    except Exception as exc:
        print(f"Failed to load CIFAR-10 dataset: {exc}")
        return

    # Normalize test data
    x_test = x_test.astype("float32") / 255.0
    y_test = y_test.reshape(-1)

    # Make predictions on test set
    print("Making predictions on test dataset...")
    y_pred_prob = model.predict(x_test, batch_size=64, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Generate confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(CLASS_NAMES)))

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True,
        cbar_kws={"label": "Count"}
    )
    plt.title("Confusion Matrix - CIFAR-10 Classification")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=100)
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    plt.close()

    # Generate classification report
    print("Generating classification report...")
    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )

    # Save classification report to file
    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("CIFAR-10 CNN Model Evaluation Report\n")
        file.write("=" * 60 + "\n\n")
        file.write(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
        file.write("Classification Report:\n")
        file.write("=" * 60 + "\n")
        file.write(report)

    print(f"Classification report saved to: {CLASSIFICATION_REPORT_PATH}")

    # Print the report to console
    print("\n" + "=" * 60)
    print("Classification Report:")
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    main()
