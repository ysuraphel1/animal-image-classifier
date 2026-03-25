import json
from pathlib import Path

from torch import nn
import matplotlib.pyplot as plt
import numpy as np

from src.config import DEVICE, METRICS_PATH
from src.data_utils import seed_everything, build_dataloaders
from src.model_utils import load_checkpoint, evaluate_model, save_metrics


def plot_confusion_matrix(confusion_matrix, class_names, output_path: Path):
    cm = np.array(confusion_matrix)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    seed_everything()

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_dataloaders()
    model, class_names, image_size = load_checkpoint()
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, y_true, y_pred = evaluate_model(model, test_loader, criterion)

    metrics = save_metrics(
        class_names=class_names,
        train_history=[],
        val_history=[],
        test_loss=test_loss,
        test_acc=test_acc,
        y_true=y_true,
        y_pred=y_pred,
    )

    confusion_png = METRICS_PATH.parent / "confusion_matrix.png"
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        metrics["class_names"],
        confusion_png,
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved confusion matrix to {confusion_png}")

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    print("Per-class metrics:")
    for class_name in class_names:
        cls_metrics = loaded["classification_report"].get(class_name, {})
        precision = cls_metrics.get("precision", 0.0)
        recall = cls_metrics.get("recall", 0.0)
        f1 = cls_metrics.get("f1-score", 0.0)
        print(
            f"{class_name}: "
            f"precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}"
        )


if __name__ == "__main__":
    main()