from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from dataset import (
    AppleLeafDataset,
    CLASS_NAMES
)

from transforms import val_test_transforms

from model import create_model


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

CSV_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_split.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "convnext"
    / "convnextv2_best_sanity.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# PRINT INFORMATION
# ============================================================

print("=" * 70)
print("VALIDATION EVALUATION")
print("=" * 70)

print()

print("Device:")
print(DEVICE)

print()

print("Model:")
print(MODEL_PATH)


# ============================================================
# DATASET
# ============================================================

validation_dataset = AppleLeafDataset(
    csv_path=CSV_PATH,
    split="validation",
    transform=val_test_transforms
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ============================================================
# LOAD MODEL
# ============================================================

model = create_model(
    num_classes=4,
    pretrained=False
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# PREDICTIONS
# ============================================================

all_labels = []
all_predictions = []

print()
print("Running validation inference...")

with torch.no_grad():

    for images, labels in validation_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(
            images
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(4)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print()
print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(4))
)

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()

print(
    pd.DataFrame(
        cm,
        index=[
            f"Actual_{x}"
            for x in CLASS_NAMES
        ],
        columns=[
            f"Pred_{x}"
            for x in CLASS_NAMES
        ]
    )
)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 8)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title(
    "ConvNeXt V2 — Validation Confusion Matrix"
)

plt.tight_layout()

confusion_path = (
    OUTPUT_DIR
    / "validation_confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE PREDICTIONS
# ============================================================

validation_results = (
    validation_dataset.data
    .copy()
)

validation_results[
    "true_index"
] = all_labels

validation_results[
    "predicted_index"
] = all_predictions

validation_results[
    "predicted_label"
] = [
        CLASS_NAMES[index]
        for index in all_predictions
    ]

validation_results[
    "correct"
] = (
    validation_results[
        "true_index"
    ]
    ==
    validation_results[
        "predicted_index"
    ]
)

predictions_path = (
    OUTPUT_DIR
    / "validation_predictions.csv"
)

validation_results.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print()

print(
    "Confusion matrix saved to:"
)

print(
    confusion_path.resolve()
)

print()

print(
    "Predictions saved to:"
)

print(
    predictions_path.resolve()
)