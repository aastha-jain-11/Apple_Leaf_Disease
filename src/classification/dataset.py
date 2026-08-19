from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_NAMES = [
    "apple_scab",
    "black_rot",
    "cedar_rust",
    "healthy"
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

INDEX_TO_CLASS = {
    index: class_name
    for class_name, index in CLASS_TO_INDEX.items()
}


# ============================================================
# DATASET
# ============================================================

class AppleLeafDataset(Dataset):

    def __init__(
        self,
        csv_path,
        split,
        transform=None
    ):

        self.csv_path = Path(csv_path).resolve()

        self.split = split

        self.transform = transform

        # ----------------------------------------------------
        # Read CSV
        # ----------------------------------------------------

        df = pd.read_csv(self.csv_path)

        # ----------------------------------------------------
        # Keep only requested split
        # ----------------------------------------------------

        self.data = df[
            df["split"] == split
        ].reset_index(drop=True)

        # ----------------------------------------------------
        # Check labels
        # ----------------------------------------------------

        unknown_labels = set(
            self.data["label"]
        ) - set(CLASS_NAMES)

        if unknown_labels:

            raise ValueError(
                f"Unknown labels found: "
                f"{unknown_labels}"
            )

        print(
            f"{split.capitalize()} dataset: "
            f"{len(self.data)} images"
        )


    def __len__(self):

        return len(self.data)


    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = Path(
            row["image_path"]
        )

        label_name = row["label"]

        label = CLASS_TO_INDEX[
            label_name
        ]

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Apply transformation
        # ----------------------------------------------------

        if self.transform is not None:

            image = self.transform(image)

        # ----------------------------------------------------
        # Convert label to tensor
        # ----------------------------------------------------

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label