"""
Unit tests for intermediate activations extraction and model layer inspection.
"""

import unittest
import numpy as np


class TestActivations(unittest.TestCase):

    def test_layer_metadata_extraction(self):
        """Verifies metadata extraction from Keras layers."""
        from src.model import build_cnn_model
        from src.activations import get_layer_metadata

        model = build_cnn_model()
        first_layer = model.layers[0]
        meta = get_layer_metadata(first_layer)

        self.assertIn("name", meta)
        self.assertIn("type", meta)
        self.assertIn("params", meta)
        self.assertIn("output_shape", meta)
        self.assertEqual(meta["type"], "Conv2D")
        self.assertEqual(meta["filters"], 32)
        self.assertEqual(meta["activation"], "relu")

    def test_extract_intermediate_activations(self):
        """Tests that real layer activations are extracted for every CNN layer."""
        from src.model import build_cnn_model, compile_model
        from src.activations import extract_intermediate_activations, get_top_active_filters

        model = build_cnn_model()
        compile_model(model)

        dummy_input = np.random.uniform(0.0, 1.0, size=(1, 28, 28, 1)).astype(np.float32)
        results = extract_intermediate_activations(model, dummy_input)

        self.assertIn("predicted_digit", results)
        self.assertIn("confidence", results)
        self.assertIn("probabilities", results)
        self.assertIn("inference_time_ms", results)
        self.assertIn("activations", results)

        acts = results["activations"]
        self.assertIn("conv2d_1", acts)
        self.assertIn("maxpool_1", acts)
        self.assertIn("conv2d_2", acts)
        self.assertIn("maxpool_2", acts)
        self.assertIn("dense_features", acts)
        self.assertIn("output_probabilities", acts)

        # Dimension checks
        self.assertEqual(acts["conv2d_1"].shape, (1, 28, 28, 32))
        self.assertEqual(acts["maxpool_1"].shape, (1, 14, 14, 32))
        self.assertEqual(acts["conv2d_2"].shape, (1, 14, 14, 64))
        self.assertEqual(acts["maxpool_2"].shape, (1, 7, 7, 64))
        self.assertEqual(acts["dense_features"].shape, (1, 128))
        self.assertEqual(acts["output_probabilities"].shape, (1, 10))

        # Test active filter selector
        top_filters = get_top_active_filters(acts["conv2d_1"], top_k=4)
        self.assertEqual(len(top_filters), 4)
        self.assertTrue(all(isinstance(f, int) and 0 <= f < 32 for f in top_filters))


if __name__ == "__main__":
    unittest.main()
