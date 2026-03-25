from torch import nn, optim

from src.config import DEVICE, NUM_EPOCHS, LEARNING_RATE, IMAGE_SIZE
from src.data_utils import seed_everything, build_dataloaders
from src.model_utils import (
    build_model,
    train_one_epoch,
    evaluate_model,
    save_checkpoint,
)


def main():
    seed_everything()

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_dataloaders()
    class_names = train_dataset.classes

    print(f"Classes: {class_names}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Using device: {DEVICE}")

    model = build_model(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, class_names, IMAGE_SIZE)
            print("Saved new best model.")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()