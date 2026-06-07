"""
Usage:
    python demo.py path/to/image.jpg
    python demo.py path/to/image.jpg --model models/best_model.pt
"""
import sys
import argparse
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from src.model import build_model
from src.dataset import CLASS_NAMES, NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD

DEFAULT_MODEL = Path(__file__).parent / 'models' / 'best_model.pt'

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    nc = checkpoint.get('num_classes', NUM_CLASSES)
    model = build_model(num_classes=nc, pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    return model


def predict(model, image_path, device):
    img = Image.open(image_path).convert('RGB')
    tensor = EVAL_TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    pred_idx = probs.argmax().item()
    return CLASS_NAMES[pred_idx], {cls: probs[i].item() for i, cls in enumerate(CLASS_NAMES)}


def _device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description='Face mask classifier inference')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--model', default=str(DEFAULT_MODEL), help='Path to model checkpoint')
    args = parser.parse_args()

    device = _device()
    model = load_model(args.model, device)

    label, probs = predict(model, args.image, device)

    print(f"\nImage  : {args.image}")
    print(f"Result : {label.upper()}")
    print("Probabilities:")
    for cls, p in sorted(probs.items(), key=lambda x: -x[1]):
        bar = '█' * int(p * 30)
        print(f"  {cls:<20} {p:.4f}  {bar}")


if __name__ == '__main__':
    main()
