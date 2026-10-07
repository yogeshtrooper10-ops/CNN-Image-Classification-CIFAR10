# CNN Image Classification using CIFAR-10
# This script loads the CIFAR-10 dataset, preprocesses it, trains a CNN model,
# plots accuracy/loss curves, and saves the trained model.

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# -----------------------------
# Fixed random seed for reproducibility
# -----------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Project paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(MODEL_DIR, "cifar10_cnn_model.keras")
TRAINING_PLOT_PATH = os.path.join(RESULTS_DIR, "training_history.png")

# Ensure required directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# -----------------------------
# Build CNN model
# -----------------------------
def build_model():
    """
    Builds a simple CNN model for CIFAR-10 image classification.
    
    Architecture:
    - 3 convolutional blocks with increasing filters (32 -> 64 -> 128)
    - Each block has Conv2D, BatchNorm, MaxPooling, and Dropout
    - Fully connected layers for classification
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        
        # Block 1: Extract low-level features
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Block 2: Extract mid-level features
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Block 3: Extract high-level features
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Classification head
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax")  # 10 CIFAR-10 classes
    ])

    # Compile with Adam optimizer and categorical crossentropy loss
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# Plot training history
# -----------------------------
def plot_training_history(history):
    """
    Saves a graph showing training and validation loss/accuracy.
    
    Args:
        history: Training history object from model.fit()
    """
    try:
        plt.figure(figsize=(12, 5))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Train Accuracy", color="blue")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="green")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Train Loss", color="blue")
        plt.plot(history.history["val_loss"], label="Validation Loss", color="green")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig(TRAINING_PLOT_PATH)
        print(f"Training graph saved to: {TRAINING_PLOT_PATH}")
    except Exception as exc:
        print(f"Error while plotting training history: {exc}")


# -----------------------------
# Main training function
# -----------------------------
def main():
    """
    Main training pipeline:
    1. Load CIFAR-10 dataset
    2. Normalize images
    3. Split into train/validation
    4. Build CNN model
    5. Train the model
    6. Save model and training plots
    """
    print("Loading CIFAR-10 dataset...")
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    except Exception as exc:
        print(f"Failed to load CIFAR-10 dataset: {exc}")
        print("Please ensure TensorFlow can download the dataset.")
        return

    # Convert to float and normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Flatten labels from shape (num_samples, 1) to (num_samples,)
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    # Split training data into train/validation (90/10 split)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.1,
        random_state=SEED,
        stratify=y_train  # Maintain class distribution
    )

    print(f"Training samples: {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")

    # Build model
    model = build_model()

    print("\nModel Architecture:")
    print(model.summary())

    # Train the model
    print("\nTraining the model...")
    try:
        history = model.fit(
            x_train,
            y_train,
            epochs=12,
            batch_size=64,
            validation_data=(x_val, y_val),
            verbose=1
        )
    except Exception as exc:
        print(f"Training failed: {exc}")
        return

    # Save trained model
    print("\nSaving trained model...")
    try:
        model.save(MODEL_PATH)
        print(f"Model saved to: {MODEL_PATH}")
    except Exception as exc:
        print(f"Failed to save model: {exc}")

    # Save training plots
    plot_training_history(history)

    # Evaluate on test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
