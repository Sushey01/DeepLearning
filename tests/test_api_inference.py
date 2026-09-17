import unittest

from src.inference_api import app


class InferenceApiTests(unittest.TestCase):
    def test_predict_route_exists(self):
        routes = [route.path for route in app.routes]
        self.assertIn('/predict', routes)

    def test_health_route_exists(self):
        routes = [route.path for route in app.routes]
        self.assertIn('/health', routes)


if __name__ == '__main__':
    unittest.main()
