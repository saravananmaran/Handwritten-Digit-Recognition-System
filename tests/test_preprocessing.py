"""
Unit tests for data preprocessing and canvas input transformations.
"""

import unittest
import numpy as np
from PIL import Image

from src.preprocessing import (
    preprocess_mnist_dataset,
    is_canvas_empty,
    preprocess_drawn_image,
    center_digit_by_mass,
    load_and_preprocess_data,
    find_ubyte_dataset_dir
)


class TestPreprocessing(unittest.TestCase):

    def test_dataset_dir_detection(self):
        """Verifies that the local MNIST dataset directory is correctly detected."""
        dataset_dir = find_ubyte_dataset_dir()
        self.assertIsNotNone(dataset_dir, "Should locate local handwritten dataset folder.")

    def test_pixel_normalization_and_shape(self):
        """Tests that raw images (0-255) are normalized to [0, 1] float32 and shaped correctly."""
        raw_images = np.random.randint(0, 256, size=(10, 28, 28), dtype=np.uint8)
        processed = preprocess_mnist_dataset(raw_images)

        self.assertEqual(processed.shape, (10, 28, 28, 1), "Output shape must be (N, 28, 28, 1)")
        self.assertEqual(processed.dtype, np.float32, "Data type must be float32")
        self.assertGreaterEqual(float(np.min(processed)), 0.0, "Minimum pixel value must be >= 0.0")
        self.assertLessEqual(float(np.max(processed)), 1.0, "Maximum pixel value must be <= 1.0")

    def test_empty_canvas_detection(self):
        """Tests detection of completely blank canvases."""
        # Pure zeros
        blank_zeros = np.zeros((280, 280, 4), dtype=np.uint8)
        self.assertTrue(is_canvas_empty(blank_zeros[:, :, :3]))

        # Attempting to preprocess empty canvas should raise ValueError
        with self.assertRaises(ValueError):
            preprocess_drawn_image(blank_zeros)

    def test_canvas_faint_noise_detection(self):
        """Faint isolated pixels should still trigger empty canvas exception."""
        faint = np.zeros((100, 100), dtype=np.float32)
        faint[10, 10] = 5.0  # tiny faint noise
        with self.assertRaises(ValueError):
            preprocess_drawn_image(faint)

    def test_drawn_digit_preprocessing(self):
        """Tests end-to-end preprocessing of a mock drawn digit canvas."""
        # Create a mock drawing: 200x200 black background with a white vertical bar (like a digit '1')
        canvas = np.zeros((200, 200, 4), dtype=np.uint8)
        canvas[:, :, 3] = 255  # full alpha
        canvas[40:160, 95:105, :3] = 255  # white stroke

        tensor, debug_img = preprocess_drawn_image(canvas)

        # Check tensor specs
        self.assertEqual(tensor.shape, (1, 28, 28, 1), "Preprocessed tensor shape must be (1, 28, 28, 1)")
        self.assertEqual(tensor.dtype, np.float32, "Preprocessed tensor dtype must be float32")
        self.assertGreaterEqual(float(np.min(tensor)), 0.0, "Min value must be >= 0.0")
        self.assertLessEqual(float(np.max(tensor)), 1.0, "Max value must be <= 1.0")

        # Check debug 28x28 image
        self.assertEqual(debug_img.shape, (28, 28), "Debug image must be 28x28")
        self.assertGreater(float(np.max(debug_img)), 0.5, "Debug image should have bright stroke pixels")

    def test_center_digit_by_mass(self):
        """Tests center-of-mass centering logic."""
        img = np.zeros((28, 28), dtype=np.float32)
        img[2:6, 2:6] = 1.0  # Corner patch
        centered = center_digit_by_mass(img)
        self.assertEqual(centered.shape, (28, 28))
        self.assertGreater(np.sum(centered), 0.0)
        # Verify it moved towards center
        cy_orig, cx_orig = 3.5, 3.5
        cy_new = np.mean(np.where(centered > 0)[0])
        cx_new = np.mean(np.where(centered > 0)[1])
        self.assertGreater(cy_new, cy_orig)
        self.assertGreater(cx_new, cx_orig)


if __name__ == "__main__":
    unittest.main()
