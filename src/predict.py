import argparse

import torch
from PIL import Image

from .data import build_transforms
from .model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    classes = ckpt["classes"]
    model = build_model(num_classes=len(classes)).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transform = build_transforms(args.image_size, train=False)
    img = Image.open(args.image).convert("RGB")
    x = transform(img).unsqueeze(0).to(args.device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    for cls, p in zip(classes, probs):
        print(f"{cls}: {p:.4f}")


if __name__ == "__main__":
    main()
