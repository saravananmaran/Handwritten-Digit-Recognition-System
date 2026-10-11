"""
Prediction pipeline for Handwritten Digit Recognition System.

Loads the trained CNN model, preprocesses the image, computes class probabilities,
and returns the predicted digit along with confidence scores.
"""

import os
import argparse
import numpy as np
from PIL import Image

from src.preprocessing import preprocess_drawn_image
from src.model import load_trained_model
from src.device import configure_device



def predict_digit(image_input, model=None, model_path: str = "models/mnist_cnn.keras") -> dict:
    """
    Performs inference on a single image (array, PIL image, or path).

    Args:
        image_input: Canvas array, PIL Image, or file path.
        model: Optional pre-loaded Keras model.
        model_path: Path to the .keras model file if model is not pre-loaded.

    Returns:
        dict containing:
            - 'predicted_digit': int (0-9)
            - 'confidence': float (0.0 to 1.0)
            - 'probabilities': list of 10 floats
            - 'processed_image': 28x28 2D numpy array used for inference
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Input image file not found: {image_input}")
        image_input = Image.open(image_input)

    # 1. Preprocess the input image into (1, 28, 28, 1) tensor
    tensor, processed_28x28 = preprocess_drawn_image(image_input)

    # 2. Load model if not provided
    if model is None:
        model = load_trained_model(model_path)

    # 3. Predict class probabilities
    raw_preds = model.predict(tensor, verbose=0)
    probabilities = raw_preds[0].astype(float).tolist()

    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    return {
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "probabilities": probabilities,
        "processed_image": processed_28x28
    }


def main():
    parser = argparse.ArgumentParser(description="Predict handwritten digit from an image file.")
    parser.add_argument("--image", type=str, required=True, help="Path to digit image file")
    parser.add_argument("--model", type=str, default="models/mnist_cnn.keras", help="Path to .keras model file")
    args = parser.parse_args()

    # Configure hardware target
    device_info = configure_device(verbose=True)

    try:
        result = predict_digit(args.image, model_path=args.model)

        print("=" * 40)
        print(" HANDWRITTEN DIGIT PREDICTION RESULT ")
        print("=" * 40)
        print(f"Predicted Digit : {result['predicted_digit']}")
        print(f"Confidence      : {result['confidence'] * 100:.2f}%")
        print("\nClass Probabilities:")
        for digit, prob in enumerate(result['probabilities']):
            bar = "#" * int(prob * 30)
            print(f"  Digit {digit}: {prob * 100:5.2f}% | {bar}")
        print("=" * 40)
    except Exception as e:
        print(f"[Error] Prediction failed: {e}")


if __name__ == "__main__":
    main()
