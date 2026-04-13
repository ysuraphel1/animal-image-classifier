import argparse
from pathlib import Path

from PIL import Image

from src.inference_utils import load_class_metrics, predict_pil_image


def main():
    parser = argparse.ArgumentParser(description="Predict animal class from an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    predictions = predict_pil_image(image, top_k=args.top_k)
    class_metrics = load_class_metrics()

    print("Predictions:")
    for label, prob in predictions:
        line = f"  {label}: confidence={prob:.2%}"
        if class_metrics and label in class_metrics:
            m = class_metrics[label]
            line += (
                f"  precision={m['precision']:.3f}"
                f"  recall={m['recall']:.3f}"
                f"  f1={m['f1']:.3f}"
            )
        print(line)


if __name__ == "__main__":
    main()