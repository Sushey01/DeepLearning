import importlib
import unittest

from src.model import build_model
from src.utils import load_config


class RequestedEnhancementTests(unittest.TestCase):
    def test_gradcam_module_exists(self):
        module = importlib.import_module("src.gradcam")
        self.assertTrue(hasattr(module, "run_gradcam"))

    def test_label_order_matches_yolo_contract(self):
        cfg = load_config("configs/config.yaml")
        self.assertEqual(cfg["classes"], ["WBC", "RBC", "Platelet"])

    def test_mobile_net_name_is_consistent(self):
        cfg = load_config("configs/config.yaml")
        self.assertIn("mobilenet_v3_small", cfg["models"])
        model = build_model("mobilenet_v3_small", num_classes=3)
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
