"""
Unit tests for device detection, memory growth configuration, and fallback management.
"""

import unittest
from src.device import configure_device, get_device_summary


class TestDeviceConfiguration(unittest.TestCase):

    def test_configure_device_structure(self):
        """Verifies that configure_device returns all expected diagnostic keys."""
        info = configure_device(verbose=False)

        self.assertIn("gpu_available", info)
        self.assertIn("device_name", info)
        self.assertIn("device_string", info)
        self.assertIn("gpu_devices", info)
        self.assertIn("memory_growth_enabled", info)
        self.assertIn("tensorflow_version", info)
        self.assertIn("status_message", info)

        self.assertIsInstance(info["gpu_available"], bool)
        self.assertIn(info["device_name"], ["GPU", "CPU"])
        self.assertIn(info["device_string"], ["/device:GPU:0", "/device:CPU:0"])

        # Consistency check: If no GPU, device_name must be CPU and device_string must be /device:CPU:0
        if not info["gpu_available"]:
            self.assertEqual(info["device_name"], "CPU")
            self.assertEqual(info["device_string"], "/device:CPU:0")
        else:
            self.assertEqual(info["device_name"], "GPU")
            self.assertEqual(info["device_string"], "/device:GPU:0")
            self.assertGreater(len(info["gpu_devices"]), 0)

    def test_get_device_summary(self):
        """Tests lightweight device summary helper for UI."""
        summary = get_device_summary()
        self.assertIn("gpu_available", summary)
        self.assertIn("device_name", summary)
        self.assertIn("tensorflow_version", summary)
        self.assertIsInstance(summary["gpu_available"], bool)


if __name__ == "__main__":
    unittest.main()
