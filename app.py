"""
Flask server for real-time face mask detection via webcam.

Usage:
    python app.py
    python app.py --model models/best_model.pt --port 5000

Then open http://localhost:5000 in your browser.
"""
import argparse
import base64
import io
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from src.model import build_model
from src.dataset import CLASS_NAMES, NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD

DEFAULT_MODEL = Path(__file__).parent / 'models' / 'best_model.pt'

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

app = Flask(__name__)
CORS(app)

_model = None
_device = None


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_model(model_path):
    global _model, _device
    _device = _get_device()
    print(f"Using device: {_device}")

    checkpoint = torch.load(model_path, map_location=_device, weights_only=True)
    nc = checkpoint.get('num_classes', NUM_CLASSES)
    _model = build_model(num_classes=nc, pretrained=False)
    _model.load_state_dict(checkpoint['state_dict'])
    _model.to(_device).eval()
    print(f"Model loaded from {model_path} ({nc} classes: {CLASS_NAMES})")


@app.route('/')
def index():
    """Serve the webcam UI."""
    with open(Path(__file__).parent / 'templates' / 'index.html', 'r') as f:
        html = f.read()
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive a base64-encoded JPEG frame, return prediction.

    Request JSON: { "image": "<base64 string>" }
    Response JSON: { "label": "with_mask", "confidence": 0.97,
                     "probabilities": {"with_mask": 0.97, "without_mask": 0.03} }
    """
    if _model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    data = request.get_json(force=True)
    img_b64 = data.get('image', '')

    # Strip data-URL prefix if present (e.g. "data:image/jpeg;base64,...")
    if ',' in img_b64:
        img_b64 = img_b64.split(',', 1)[1]

    try:
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Invalid image: {e}'}), 400

    tensor = EVAL_TRANSFORM(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    label = CLASS_NAMES[pred_idx]
    probabilities = {cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)}

    return jsonify({
        'label': label,
        'confidence': float(probs[pred_idx]),
        'probabilities': probabilities,
    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': _model is not None})


def main():
    parser = argparse.ArgumentParser(description='Face mask detection web server')
    parser.add_argument('--model', default=str(DEFAULT_MODEL),
                        help='Path to model checkpoint (default: models/best_model.pt)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on (default: 5000)')
    parser.add_argument('--host', default='127.0.0.1',
                        help='Host to bind to (default: 127.0.0.1)')
    args = parser.parse_args()

    load_model(args.model)
    print(f"\nServer running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()