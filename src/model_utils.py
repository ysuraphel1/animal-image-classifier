import json
from typing import Dict, List

import torch
from torch import nn
from torchvision import models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import DEVICE, MODEL_PATH, CLASS_NAMES_PATH, METRICS_PATH


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    y_true = []
    y_pred = []

    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    avg_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc


def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    avg_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc, y_true, y_pred


def save_checkpoint(model, class_names: List[str], image_size: int) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": image_size,
    }
    torch.save(checkpoint, MODEL_PATH)

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)


def load_checkpoint():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_PATH}. Run training first."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 224)

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names, image_size


def save_metrics(
    class_names: List[str],
    train_history: List[Dict],
    val_history: List[Dict],
    test_loss: float,
    test_acc: float,
    y_true: List[int],
    y_pred: List[int],
) -> Dict:
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "train_history": train_history,
        "val_history": val_history,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "class_names": class_names,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics