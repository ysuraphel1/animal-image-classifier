import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.config import (
    RAW_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMAGE_EXTENSIONS,
    RANDOM_SEED,
    IMAGE_SIZE,
    BATCH_SIZE,
)


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_class_directories(raw_dir: Path = RAW_DIR) -> List[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    class_dirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found under {raw_dir}")

    return sorted(class_dirs)


def collect_images_by_class(raw_dir: Path = RAW_DIR) -> Dict[str, List[Path]]:
    class_to_images: Dict[str, List[Path]] = {}

    for class_dir in list_class_directories(raw_dir):
        images = [p for p in class_dir.rglob("*") if is_image_file(p)]
        if not images:
            raise ValueError(f"No images found in class folder: {class_dir}")
        class_to_images[class_dir.name] = sorted(images)

    return class_to_images


def reset_split_dirs() -> None:
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def split_class_images(
    image_paths: List[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[Path], List[Path], List[Path]]:
    total = len(image_paths)
    if total < 3:
        raise ValueError(
            "Each class should have at least 3 images to create train/val/test splits."
        )

    shuffled = image_paths[:]
    random.shuffle(shuffled)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = shuffled[:train_end]
    val_files = shuffled[train_end:val_end]
    test_files = shuffled[val_end:]

    if not train_files or not val_files or not test_files:
        raise ValueError(
            "A class ended up with an empty split. Add more images per class."
        )

    return train_files, val_files, test_files


def copy_split_files(class_name: str, files: List[Path], destination_root: Path) -> None:
    class_dest = destination_root / class_name
    class_dest.mkdir(parents=True, exist_ok=True)

    for idx, file_path in enumerate(files):
        ext = file_path.suffix.lower()
        destination = class_dest / f"{class_name}_{idx:05d}{ext}"
        shutil.copy2(file_path, destination)


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, eval_transform


def build_dataloaders():
    train_transform, eval_transform = build_transforms()

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=eval_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader