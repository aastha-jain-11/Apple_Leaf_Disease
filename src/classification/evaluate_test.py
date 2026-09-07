from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

from dataset import AppleLeafDataset, CLASS_NAMES
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
    / "ablation"
    / "C_FULL"
    / "best_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "test"
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
print("FINAL TEST EVALUATION")
print("=" * 70)

print()
print("Device:")
print(DEVICE)

if torch.cuda.is_available():
    print()
    print("GPU:")
    print(torch.cuda.get_device_name(0))

print()
print("Model:")
print(MODEL_PATH)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found:\n{MODEL_PATH}"
    )


# ============================================================
# DATASET
# ============================================================

test_dataset = AppleLeafDataset(
    csv_path=CSV_PATH,
    split="test",
    transform=val_test_transforms
)

print()
print("Test dataset:")
print(f"{len(test_dataset)} images")


# ============================================================
# DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

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
# TEST INFERENCE
# ============================================================

all_labels = []
all_predictions = []

print()
print("Running test inference...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)


# ============================================================
# BASIC CHECK
# ============================================================

print()
print("=" * 70)
print("TEST SET CHECK")
print("=" * 70)

print()
print("Number of test samples:")
print(len(all_labels))

print()
print("Number of predictions:")
print(len(all_predictions))


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(4)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
    output_dict=True
)

report_text = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(4)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_precision = report_dict["macro avg"]["precision"]
macro_recall = report_dict["macro avg"]["recall"]
macro_f1 = report_dict["macro avg"]["f1-score"]


print()
print("=" * 70)
print("FINAL TEST CLASSIFICATION REPORT")
print("=" * 70)

print()
print(report_text)


# ============================================================
# SUMMARY METRICS
# ============================================================

print()
print("=" * 70)
print("SUMMARY METRICS")
print("=" * 70)

print()
print(f"Accuracy:          {accuracy:.4f}")
print(f"Macro Precision:   {macro_precision:.4f}")
print(f"Macro Recall:      {macro_recall:.4f}")
print(f"Macro F1:          {macro_f1:.4f}")


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
# SAVE CONFUSION MATRIX
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
    "ConvNeXt V2 — Final Test Confusion Matrix"
)

plt.tight_layout()

confusion_path = (
    OUTPUT_DIR
    / "test_confusion_matrix.png"
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

test_results = (
    test_dataset.data
    .copy()
)

test_results["true_index"] = all_labels

test_results["predicted_index"] = all_predictions

test_results["predicted_label"] = [
    CLASS_NAMES[index]
    for index in all_predictions
]

test_results["correct"] = (
    test_results["true_index"]
    ==
    test_results["predicted_index"]
)

predictions_path = (
    OUTPUT_DIR
    / "test_predictions.csv"
)

test_results.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "experiment": "C_FULL",
    "model": "ConvNeXt V2 Base",
    "test_samples": int(len(all_labels)),
    "accuracy": float(accuracy),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
}

for class_name in CLASS_NAMES:
    metrics[class_name] = {
        "precision": float(
            report_dict[class_name]["precision"]
        ),
        "recall": float(
            report_dict[class_name]["recall"]
        ),
        "f1": float(
            report_dict[class_name]["f1-score"]
        ),
        "support": int(
            report_dict[class_name]["support"]
        )
    }

metrics_path = (
    OUTPUT_DIR
    / "test_metrics.json"
)

import json

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("TEST EVALUATION COMPLETE")
print("=" * 70)

print()
print("Confusion matrix saved to:")
print(confusion_path.resolve())

print()
print("Predictions saved to:")
print(predictions_path.resolve())

print()
print("Metrics saved to:")
print(metrics_path.resolve())