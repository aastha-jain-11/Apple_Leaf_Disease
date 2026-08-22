from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from dataset import (
    AppleLeafDataset,
    CLASS_NAMES
)

from transforms import (
    train_transforms,
    val_test_transforms
)

from model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_split.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "convnext"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

RANDOM_SEED = 42

BATCH_SIZE = 32

NUM_WORKERS = 0

NUM_CLASSES = 4

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_EPOCHS = 2

USE_AMP = True


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("CONVNEXT V2 TRAINING")
print("=" * 70)

print()

print("Project root:")
print(PROJECT_ROOT)

print()

print("Device:")
print(device)

if device.type == "cuda":

    print()

    print("GPU:")
    print(
        torch.cuda.get_device_name(0)
    )

    print()

    print("GPU memory:")
    print(
        f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
    )


# ============================================================
# DATASETS
# ============================================================

print()
print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

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


# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

print()
print("=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)

train_labels = (
    train_dataset.data["label"]
)

class_counts = (
    train_labels
    .value_counts()
    .reindex(CLASS_NAMES)
)

print()

print("Training class counts:")

print(
    class_counts.to_string()
)


# ------------------------------------------------------------
# Inverse-frequency weighting
# ------------------------------------------------------------

total_samples = (
    class_counts.sum()
)

class_weights = (
    total_samples
    /
    (
        NUM_CLASSES
        * class_counts
    )
)

print()

print("Class weights:")

for class_name, weight in zip(
    CLASS_NAMES,
    class_weights
):

    print(
        f"{class_name:15s}: "
        f"{weight:.4f}"
    )


class_weights_tensor = torch.tensor(
    class_weights.values,
    dtype=torch.float32,
    device=device
)


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("=" * 70)
print("CREATING MODEL")
print("=" * 70)

model = create_model(
    num_classes=NUM_CLASSES,
    pretrained=True
)

model = model.to(device)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# MIXED PRECISION
# ============================================================

if USE_AMP and device.type == "cuda":

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# METRICS FUNCTION
# ============================================================

def calculate_metrics(
    true_labels,
    predicted_labels
):

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    all_labels = []
    all_predictions = []

    start_time = time.time()

    for batch_index, (
        images,
        labels
    ) in enumerate(train_loader):

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        if scaler is not None:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )

            # ------------------------------------------------
            # Backward pass
            # ------------------------------------------------

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index + 1
        ) % 10 == 0:

            print(
                f"  Batch "
                f"{batch_index + 1:3d}/"
                f"{len(train_loader)} "
                f"| Loss: {loss.item():.4f}"
            )


    epoch_loss = (
        running_loss
        /
        len(train_dataset)
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    elapsed = (
        time.time()
        - start_time
    )

    return (
        epoch_loss,
        metrics,
        elapsed
    )


# ============================================================
# VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []

    for images, labels in validation_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        if (
            device.type == "cuda"
            and USE_AMP
        ):

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )

        else:

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )


        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


    epoch_loss = (
        running_loss
        /
        len(validation_dataset)
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    return (
        epoch_loss,
        metrics
    )


# ============================================================
# TRAINING LOOP
# ============================================================

history = []

best_val_f1 = -1.0

best_model_path = (
    MODEL_DIR
    / "convnextv2_best_sanity.pth"
)


print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print()

print(
    f"Epochs: {NUM_EPOCHS}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)

print(
    f"AMP enabled: {USE_AMP}"
)


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    print()
    print("=" * 70)

    print(
        f"EPOCH {epoch}/{NUM_EPOCHS}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_loss, train_metrics, train_time = (
        train_one_epoch()
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_loss, val_metrics = (
        validate()
    )


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()

    print("TRAIN")

    print(
        f"Loss:      {train_loss:.4f}"
    )

    print(
        f"Accuracy:  "
        f"{train_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1:  "
        f"{train_metrics['macro_f1']:.4f}"
    )

    print()

    print("VALIDATION")

    print(
        f"Loss:      {val_loss:.4f}"
    )

    print(
        f"Accuracy:  "
        f"{val_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{val_metrics['macro_precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{val_metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1:  "
        f"{val_metrics['macro_f1']:.4f}"
    )

    print()

    print(
        f"Epoch time: "
        f"{train_time:.1f} seconds"
    )


    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_entry = {

        "epoch": epoch,

        "train_loss": train_loss,

        "train_accuracy":
            train_metrics["accuracy"],

        "train_macro_f1":
            train_metrics["macro_f1"],

        "val_loss": val_loss,

        "val_accuracy":
            val_metrics["accuracy"],

        "val_macro_precision":
            val_metrics["macro_precision"],

        "val_macro_recall":
            val_metrics["macro_recall"],

        "val_macro_f1":
            val_metrics["macro_f1"],

        "epoch_time_seconds":
            train_time
    }

    history.append(
        history_entry
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if (
        val_metrics["macro_f1"]
        > best_val_f1
    ):

        best_val_f1 = (
            val_metrics["macro_f1"]
        )

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_macro_f1":
                    best_val_f1,

                "class_names":
                    CLASS_NAMES,

                "config": {
                    "model_name":
                        "convnextv2_base.fcmae_ft_in1k",

                    "num_classes":
                        NUM_CLASSES,

                    "image_size":
                        224,

                    "random_seed":
                        RANDOM_SEED
                }
            },
            best_model_path
        )

        print()

        print(
            "New best model saved!"
        )

        print(
            best_model_path
        )


# ============================================================
# SAVE HISTORY
# ============================================================

history_path = (
    OUTPUT_DIR
    / "sanity_training_history.json"
)

with open(
    history_path,
    "w"
) as file:

    json.dump(
        history,
        file,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("SANITY TRAINING COMPLETE")
print("=" * 70)

print()

print(
    "Best validation Macro F1:"
)

print(
    f"{best_val_f1:.4f}"
)

print()

print(
    "Best model:"
)

print(
    best_model_path.resolve()
)

print()

print(
    "Training history:"
)

print(
    history_path.resolve()
)