from pathlib import Path
import argparse
import json
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms as tv_transforms

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
    val_test_transforms,
    IMAGE_SIZE
)

from model import create_model


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================

EXPERIMENTS = {

    "C_FULL": {
        "description": "Full proposed classification configuration",
        "pretrained": True,
        "augmentation": True,
        "class_weights": True,
        "scheduler": True
    },

    "C_NO_AUG": {
        "description": "Ablation without data augmentation",
        "pretrained": True,
        "augmentation": False,
        "class_weights": True,
        "scheduler": True
    },

    "C_NO_WEIGHTS": {
        "description": "Ablation without class weighting",
        "pretrained": True,
        "augmentation": True,
        "class_weights": False,
        "scheduler": True
    },

    "C_SCRATCH": {
        "description": "Ablation using random initialization",
        "pretrained": False,
        "augmentation": True,
        "class_weights": True,
        "scheduler": True
    },

    "C_NO_SCHEDULER": {
        "description": "Ablation without learning-rate scheduler",
        "pretrained": True,
        "augmentation": True,
        "class_weights": True,
        "scheduler": False
    }
}


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="ConvNeXt V2 classification experiment runner"
)

parser.add_argument(
    "--experiment",
    type=str,
    required=True,
    choices=list(EXPERIMENTS.keys()),
    help="Experiment configuration to run"
)

args = parser.parse_args()

EXPERIMENT_NAME = args.experiment
EXPERIMENT_CONFIG = EXPERIMENTS[EXPERIMENT_NAME]


# ============================================================
# GENERAL CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "ablation"
)

EXPERIMENT_OUTPUT_DIR = (
    OUTPUT_ROOT
    / EXPERIMENT_NAME
)

EXPERIMENT_MODEL_DIR = (
    MODEL_DIR
    / "ablation"
    / EXPERIMENT_NAME
)


# ============================================================
# HYPERPARAMETERS
# ============================================================

RANDOM_SEED = 42

BATCH_SIZE = 32

NUM_WORKERS = 0

NUM_CLASSES = 4

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_EPOCHS = 15

USE_AMP = True

EARLY_STOPPING_PATIENCE = 5


# ============================================================
# CREATE DIRECTORIES
# ============================================================

EXPERIMENT_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EXPERIMENT_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )

    torch.backends.cudnn.benchmark = True


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# PRINT EXPERIMENT INFORMATION
# ============================================================

print("=" * 70)
print("CONVNEXT V2 CLASSIFICATION ABLATION")
print("=" * 70)

print()

print("Experiment:")
print(EXPERIMENT_NAME)

print()

print("Description:")
print(EXPERIMENT_CONFIG["description"])

print()

print("Configuration:")
print(
    json.dumps(
        EXPERIMENT_CONFIG,
        indent=4
    )
)

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

# ------------------------------------------------------------
# Training transforms
# ------------------------------------------------------------

if EXPERIMENT_CONFIG["augmentation"]:

    selected_train_transforms = train_transforms

else:

    selected_train_transforms = tv_transforms.Compose([

        tv_transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        tv_transforms.ToTensor(),

        tv_transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])


train_dataset = AppleLeafDataset(
    csv_path=CSV_PATH,
    split="train",
    transform=selected_train_transforms
)


validation_dataset = AppleLeafDataset(
    csv_path=CSV_PATH,
    split="validation",
    transform=val_test_transforms
)


print()

# print(
#     f"Train dataset: "
#     f"{len(train_dataset)} images"
# )

# print(
#     f"Validation dataset: "
#     f"{len(validation_dataset)} images"
# )


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
# CLASS WEIGHTS
# ============================================================

print()
print("=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)

train_labels = train_dataset.data["label"]

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


if EXPERIMENT_CONFIG["class_weights"]:

    total_samples = class_counts.sum()

    class_weights = (
        total_samples
        /
        (
            NUM_CLASSES
            * class_counts
        )
    )

    print()

    print("Class weighting: ENABLED")

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

else:

    print()

    print("Class weighting: DISABLED")

    class_weights_tensor = None


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("=" * 70)
print("CREATING MODEL")
print("=" * 70)

model = create_model(
    num_classes=NUM_CLASSES,
    pretrained=EXPERIMENT_CONFIG["pretrained"]
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
# LEARNING RATE SCHEDULER
# ============================================================

if EXPERIMENT_CONFIG["scheduler"]:

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )

    print()
    print("Scheduler: CosineAnnealingLR")

else:

    scheduler = None

    print()
    print("Scheduler: DISABLED")


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
# METRICS
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
        # Forward + backward
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
# SAVE EXPERIMENT CONFIGURATION
# ============================================================

full_config = {

    "experiment": EXPERIMENT_NAME,

    "experiment_description":
        EXPERIMENT_CONFIG["description"],

    "experiment_settings":
        EXPERIMENT_CONFIG,

    "hyperparameters": {

        "random_seed":
            RANDOM_SEED,

        "batch_size":
            BATCH_SIZE,

        "num_workers":
            NUM_WORKERS,

        "num_classes":
            NUM_CLASSES,

        "image_size":
            IMAGE_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "num_epochs":
            NUM_EPOCHS,

        "use_amp":
            USE_AMP,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE
    },

    "dataset": {

        "csv":
            str(CSV_PATH),

        "train_size":
            len(train_dataset),

        "validation_size":
            len(validation_dataset),

        "classes":
            CLASS_NAMES
    }
}


config_path = (
    EXPERIMENT_OUTPUT_DIR
    / "config.json"
)

with open(
    config_path,
    "w"
) as file:

    json.dump(
        full_config,
        file,
        indent=4
    )


# ============================================================
# TRAINING LOOP
# ============================================================

history = []

best_val_f1 = -1.0

best_epoch = 0

epochs_without_improvement = 0


best_model_path = (
    EXPERIMENT_MODEL_DIR
    / "best_model.pth"
)


print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print()

print(
    f"Experiment: {EXPERIMENT_NAME}"
)

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
    f"Weight decay: {WEIGHT_DECAY}"
)

print(
    f"AMP enabled: {USE_AMP}"
)

print(
    f"Pretrained: "
    f"{EXPERIMENT_CONFIG['pretrained']}"
)

print(
    f"Augmentation: "
    f"{EXPERIMENT_CONFIG['augmentation']}"
)

print(
    f"Class weights: "
    f"{EXPERIMENT_CONFIG['class_weights']}"
)

print(
    f"Scheduler: "
    f"{EXPERIMENT_CONFIG['scheduler']}"
)


# ============================================================
# EPOCHS
# ============================================================

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

    (
        train_loss,
        train_metrics,
        train_time
    ) = train_one_epoch()


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    (
        val_loss,
        val_metrics
    ) = validate()


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    if scheduler is not None:

        scheduler.step()


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()

    print("TRAIN")

    print(
        f"Loss:      "
        f"{train_loss:.4f}"
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
        f"Loss:      "
        f"{val_loss:.4f}"
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
        f"Learning rate: "
        f"{current_lr:.8f}"
    )

    print(
        f"Epoch time: "
        f"{train_time:.1f} seconds"
    )


    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history_entry = {

        "epoch":
            epoch,

        "learning_rate":
            current_lr,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_metrics["accuracy"],

        "train_macro_f1":
            train_metrics["macro_f1"],

        "val_loss":
            val_loss,

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
    # Best model
    # --------------------------------------------------------

    if (
        val_metrics["macro_f1"]
        > best_val_f1
    ):

        best_val_f1 = (
            val_metrics["macro_f1"]
        )

        best_epoch = epoch

        epochs_without_improvement = 0


        torch.save(
            {

                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_macro_f1":
                    best_val_f1,

                "class_names":
                    CLASS_NAMES,

                "experiment":
                    EXPERIMENT_NAME,

                "config":
                    full_config
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

    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print()

        print(
            "Early stopping triggered."
        )

        print(
            f"No validation Macro F1 "
            f"improvement for "
            f"{EARLY_STOPPING_PATIENCE} epochs."
        )

        break


# ============================================================
# SAVE HISTORY
# ============================================================

history_path = (
    EXPERIMENT_OUTPUT_DIR
    / "training_history.json"
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
# SAVE FINAL RESULT
# ============================================================

best_result = {

    "experiment":
        EXPERIMENT_NAME,

    "best_epoch":
        best_epoch,

    "best_validation_macro_f1":
        best_val_f1,

    "best_validation_accuracy":
        max(
            item["val_accuracy"]
            for item in history
        ),

    "configuration":
        EXPERIMENT_CONFIG,

    "hyperparameters": {

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "num_epochs":
            NUM_EPOCHS,

        "random_seed":
            RANDOM_SEED
    }
}


result_path = (
    EXPERIMENT_OUTPUT_DIR
    / "result.json"
)

with open(
    result_path,
    "w"
) as file:

    json.dump(
        best_result,
        file,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

print()

print(
    f"Experiment: "
    f"{EXPERIMENT_NAME}"
)

print()

print(
    f"Best validation Macro F1: "
    f"{best_val_f1:.4f}"
)

print()

print(
    f"Best epoch: "
    f"{best_epoch}"
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
    "Configuration:"
)

print(
    config_path.resolve()
)

print()

print(
    "Training history:"
)

print(
    history_path.resolve()
)

print()

print(
    "Result:"
)

print(
    result_path.resolve()
)