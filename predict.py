# Prediction script for CIFAR-10 CNN model.
# It allows prediction on a custom image or on a sample from CIFAR-10.
#
# Usage:
#   python predict.py --image path/to/image.jpg
#   python predict.py

import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
# Project paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(MODEL_DIR, "cifar10_cnn_model.keras")
PREDICTION_OUTPUT_PATH = os.path.join(RESULTS_DIR, "prediction_result.png")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# -----------------------------
# Load model
# -----------------------------
def load_model():
    """
    Loads the trained CNN model from disk.
    
    Returns:
        Loaded Keras model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please train the model first with train.py."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from: {MODEL_PATH}")
    return model


# -----------------------------
# Preprocess image
# -----------------------------
def preprocess_image(image_path):
    """
    Loads and preprocesses an image for prediction.
    
    Args:
        image_path: Path to image file (JPG, PNG, etc.)
        
    Returns:
        Tuple of (preprocessed_array, original_image)
        
    Raises:
        ValueError: If image cannot be opened
    """
    try:
        # Load image and convert to RGB
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not open image '{image_path}': {exc}")

    # Resize to CIFAR-10 input size (32x32)
    image = image.resize((32, 32))

    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image, dtype=np.float32) / 255.0

    # Add batch dimension for Keras (required for batch prediction)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array, image


# -----------------------------
# Predict from custom image or CIFAR sample
# -----------------------------
def predict_from_image(model, image_path=None):
    """
    Makes a prediction on an image (custom or sample CIFAR-10).
    
    Args:
        model: Loaded Keras model
        image_path: Optional path to custom image. If None, uses CIFAR-10 sample.
    """
    if image_path is not None:
        # Load custom image
        image_array, original_image = preprocess_image(image_path)
        true_label = None
        print(f"Predicting class for: {image_path}")
    else:
        # Use a sample from CIFAR-10 test set
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        image_array = x_test[0:1].astype("float32") / 255.0
        original_image = Image.fromarray((x_test[0] * 255).astype("uint8"))
        true_label = int(y_test[0][0])
        print("No image path provided. Using a sample CIFAR-10 image.")

    # Make prediction
    print("Making prediction...")
    prediction_prob = model.predict(image_array, verbose=0)
    predicted_index = int(np.argmax(prediction_prob, axis=1)[0])
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(prediction_prob[0][predicted_index])

    # Display results
    if true_label is not None:
        true_name = CLASS_NAMES[true_label]
        print(f"Actual class: {true_name}")
    print(f"Predicted class: {predicted_label}")
    print(f"Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")

    # Display prediction probabilities for all classes
    print("\nPrediction probabilities:")
    for idx, prob in enumerate(prediction_prob[0]):
        prob_value = float(prob)
        bar = "█" * int(prob_value * 30)
        print(f"  {CLASS_NAMES[idx]:12s}: {prob_value:.4f} {bar}")

    # Save visualization
    print("\nSaving prediction visualization...")
    plt.figure(figsize=(6, 6))
    plt.imshow(original_image)
    title = f"Predicted: {predicted_label} ({confidence * 100:.1f}%)"
    if true_label is not None:
        title += f"\nActual: {CLASS_NAMES[true_label]}"
    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(PREDICTION_OUTPUT_PATH, bbox_inches="tight", dpi=100)
    print(f"Prediction visualization saved to: {PREDICTION_OUTPUT_PATH}")
    plt.close()


def main():
    """
    Main entry point for prediction script.
    Handles command-line arguments and coordinates prediction.
    """
    parser = argparse.ArgumentParser(
        description="Predict CIFAR-10 class for an image using trained CNN."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to a custom image file (JPG/PNG/BMP/GIF)."
    )
    args = parser.parse_args()

    try:
        model = load_model()
        predict_from_image(model, args.image)
        print("\nPrediction completed successfully!")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
    except ValueError as exc:
        print(f"Error: {exc}")
    except Exception as exc:
        print(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()
