from src.config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
from src.data_utils import (
    seed_everything,
    collect_images_by_class,
    reset_split_dirs,
    split_class_images,
    copy_split_files,
)
from src.config import TRAIN_DIR, VAL_DIR, TEST_DIR


def main():
    seed_everything()

    if round(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT, 5) != 1.0:
        raise ValueError("TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT must equal 1.0")

    class_to_images = collect_images_by_class()
    reset_split_dirs()

    print("Preparing dataset splits...")

    for class_name, image_paths in class_to_images.items():
        train_files, val_files, test_files = split_class_images(
            image_paths=image_paths,
            train_ratio=TRAIN_SPLIT,
            val_ratio=VAL_SPLIT,
            test_ratio=TEST_SPLIT,
        )

        copy_split_files(class_name, train_files, TRAIN_DIR)
        copy_split_files(class_name, val_files, VAL_DIR)
        copy_split_files(class_name, test_files, TEST_DIR)

        print(
            f"{class_name}: "
            f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
        )

    print("Done. Dataset split folders created under data/train, data/val, data/test.")


if __name__ == "__main__":
    main()