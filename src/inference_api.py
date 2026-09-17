import sys
from pathlib import Path

from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import load_model, predict_image_bytes

app = Flask(__name__)

MODEL = None
CLASSES = None
CFG = None
DEVICE = None


def setup_model(model_name: str = "efficientnet_b0", config_path: str = "configs/config.yaml"):
    global MODEL, CLASSES, CFG, DEVICE
    MODEL, CLASSES, CFG, DEVICE = load_model(model_name, config_path)


@app.get('/health')
def health():
    return jsonify({"status": "ok", "model": "efficientnet_b0"})


@app.post('/predict')
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if MODEL is None:
        setup_model()

    image_bytes = file.read()
    result = predict_image_bytes(
        image_bytes,
        MODEL,
        CLASSES,
        CFG,
        DEVICE,
        threshold=0.0,
        image_name=file.filename,
    )
    return jsonify({
        "predicted_class": result["predicted_class"],
        "confidence": round(result["confidence"], 4),
        "probabilities": {k: round(v, 4) for k, v in result["probabilities"].items()},
    })


if __name__ == '__main__':
    setup_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
