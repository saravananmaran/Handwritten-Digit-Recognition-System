"""
Unit tests for model architecture, prediction output shapes, probabilities, and ranges.
"""

import unittest
import numpy as np


class TestModelAndPrediction(unittest.TestCase):

    def test_cnn_architecture_shapes(self):
        """Verifies CNN layer dimensions and output layer configuration."""
        import tensorflow as tf
        from src.model import build_cnn_model

        model = build_cnn_model(input_shape=(28, 28, 1), num_classes=10)

        # Verify output layer shape
        self.assertEqual(model.output_shape, (None, 10), "Model output shape must be (batch_size, 10)")

        # Verify output activation is softmax
        output_layer = model.layers[-1]
        self.assertEqual(output_layer.activation.__name__, "softmax", "Output layer activation must be softmax")

    def test_prediction_probabilities_and_ranges(self):
        """Tests that model inference outputs valid probabilities summing to ~1.0."""
        import tensorflow as tf
        from src.model import build_cnn_model, compile_model

        model = build_cnn_model()
        compile_model(model)

        # Create dummy batch of 3 samples
        dummy_inputs = np.random.uniform(0.0, 1.0, size=(3, 28, 28, 1)).astype(np.float32)
        preds = model.predict(dummy_inputs, verbose=0)

        self.assertEqual(preds.shape, (3, 10), "Batch prediction shape must be (3, 10)")

        for row in preds:
            # Each probability must be between 0 and 1
            self.assertTrue(np.all(row >= 0.0) and np.all(row <= 1.0), "All probabilities must be in [0, 1]")
            # Sum of probabilities must equal 1.0 within floating point precision
            prob_sum = float(np.sum(row))
            self.assertAlmostEqual(prob_sum, 1.0, places=5, msg="Probabilities must sum to 1.0")
            # Predicted class must be in 0-9
            pred_class = int(np.argmax(row))
            self.assertIn(pred_class, list(range(10)), "Predicted class must be an integer between 0 and 9")

    def test_predict_digit_pipeline_with_canvas(self):
        """Tests predict_digit function with a mock canvas drawing."""
        from src.model import build_cnn_model, compile_model
        from predict import predict_digit

        model = build_cnn_model()
        compile_model(model)

        # Mock drawing canvas with stroke
        canvas = np.zeros((150, 150, 4), dtype=np.uint8)
        canvas[:, :, 3] = 255
        canvas[30:120, 70:80, :3] = 255  # vertical stroke

        result = predict_digit(canvas, model=model)

        self.assertIn("predicted_digit", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertIn("processed_image", result)

        self.assertIsInstance(result["predicted_digit"], int)
        self.assertIn(result["predicted_digit"], range(10))
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertEqual(len(result["probabilities"]), 10)
        self.assertAlmostEqual(sum(result["probabilities"]), 1.0, places=4)
        self.assertEqual(result["processed_image"].shape, (28, 28))


if __name__ == "__main__":
    unittest.main()
