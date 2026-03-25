from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms

from src.config import DEVICE
from src.model_utils import load_checkpoint


def get_inference_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def predict_pil_image(image: Image.Image, top_k: int = 3) -> List[Tuple[str, float]]:
    model, class_names, image_size = load_checkpoint()
    transform = get_inference_transform(image_size)

    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    k = min(top_k, len(class_names))
    top_probs, top_indices = torch.topk(probabilities, k=k)

    results = []
    for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
        results.append((class_names[idx], float(prob)))

    return results