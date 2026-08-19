from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import AppleLeafDataset
from transforms import (
    train_transforms,
    val_test_transforms
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "outputs" / "dataset_split.csv"

BATCH_SIZE = 32

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("DATALOADER TEST")
print("=" * 70)

print()
print("Device:", device)


# ============================================================
# DATASETS
# ============================================================

train_dataset = AppleLeafDataset(
    csv_path=CSV_PATH,
    split="train",
    transform=train_transforms
)

validation_dataset = AppleLeafDataset(
    csv_path=CSV_PATH,
    split="validation",
    transform=val_test_transforms
)

test_dataset = AppleLeafDataset(
    csv_path=CSV_PATH,
    split="test",
    transform=val_test_transforms
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ============================================================
# GET ONE BATCH
# ============================================================

images, labels = next(
    iter(train_loader)
)


print()
print("CPU batch obtained successfully.")

print(
    "Image tensor shape:",
    images.shape
)

print(
    "Label tensor shape:",
    labels.shape
)

print(
    "Labels:",
    labels
)

print(
    "Image dtype:",
    images.dtype
)


# ============================================================
# MOVE TO GPU
# ============================================================

print()
print("Moving batch to GPU...")

images = images.to(
    device,
    non_blocking=True
)

labels = labels.to(
    device,
    non_blocking=True
)

torch.cuda.synchronize()


print()
print("Images device:")
print(images.device)

print()

print("Labels device:")
print(labels.device)

print()

print("=" * 70)
print("DATALOADER + GPU TEST PASSED")
print("=" * 70)