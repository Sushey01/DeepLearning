import io
import json
import unittest

from PIL import Image

from src.inference_api import app
from src.reporting import build_model_summary


class EnhancementFeatureTests(unittest.TestCase):
    def test_api_threshold_is_exposed(self):
        client = app.test_client()

        image = Image.new("RGB", (224, 224), color="red")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        response = client.post(
            "/predict",
            data={
                "file": (buffer, "sample.png"),
                "threshold": "0.9",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("threshold", payload)
        self.assertEqual(payload["threshold"], 0.9)

    def test_model_summary_has_best_model_and_table(self):
        metrics = {
            "efficientnet_b0": {"accuracy": 0.99, "macro_f1": 0.985, "params": 5000000, "train_time_min": 8.2},
            "mobilenet_v3_small": {"accuracy": 0.97, "macro_f1": 0.96, "params": 2200000, "train_time_min": 6.0},
            "densenet121": {"accuracy": 0.98, "macro_f1": 0.97, "params": 7000000, "train_time_min": 12.0},
        }

        markdown = build_model_summary(metrics)
        self.assertIn("Best model", markdown)
        self.assertIn("| Model | Accuracy | Macro F1 |", markdown)
        self.assertIn("efficientnet_b0", markdown)


if __name__ == "__main__":
    unittest.main()
