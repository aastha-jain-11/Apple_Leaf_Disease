from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = Path("outputs/dataset_split.csv")
OUTPUT_DIR = Path("outputs")

RANDOM_SEED = 42
IMAGES_PER_CLASS = 5


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("VISUAL DATASET INSPECTION")
print("=" * 70)

print()

print(f"Total images: {len(df)}")

print()

print("Classes:")
print(
    df["label"]
    .value_counts()
    .to_string()
)

print()


# ============================================================
# CREATE ONE GRID FOR EACH CLASS
# ============================================================

classes = sorted(df["label"].unique())

for class_name in classes:

    class_df = df[
        df["label"] == class_name
    ]

    samples = class_df.sample(
        n=min(IMAGES_PER_CLASS, len(class_df)),
        random_state=RANDOM_SEED
    )

    fig, axes = plt.subplots(
        1,
        len(samples),
        figsize=(15, 4)
    )

    # Handle the case of one image
    if len(samples) == 1:
        axes = [axes]

    for ax, (_, row) in zip(
        axes,
        samples.iterrows()
    ):

        image_path = Path(row["image_path"])

        try:

            image = Image.open(image_path)

            ax.imshow(image)

            ax.set_title(
                f"{class_name}\n"
                f"{image_path.name}",
                fontsize=8
            )

            ax.axis("off")

        except Exception as error:

            ax.set_title(
                f"ERROR\n{error}",
                fontsize=8
            )

            ax.axis("off")

    plt.suptitle(
        f"Random Samples — {class_name}",
        fontsize=14
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        f"samples_{class_name}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# CREATE COMBINED GRID
# ============================================================

fig, axes = plt.subplots(
    len(classes),
    IMAGES_PER_CLASS,
    figsize=(15, 12)
)

for row_index, class_name in enumerate(classes):

    class_df = df[
        df["label"] == class_name
    ]

    samples = class_df.sample(
        n=min(IMAGES_PER_CLASS, len(class_df)),
        random_state=RANDOM_SEED
    )

    for col_index in range(IMAGES_PER_CLASS):

        ax = axes[row_index, col_index]

        if col_index >= len(samples):

            ax.axis("off")
            continue

        row = samples.iloc[col_index]

        image_path = Path(row["image_path"])

        try:

            image = Image.open(image_path)

            ax.imshow(image)

            ax.set_title(
                class_name,
                fontsize=9
            )

        except Exception:

            ax.set_title(
                "ERROR"
            )

        ax.axis("off")


plt.suptitle(
    "Apple Leaf Dataset — Random Samples",
    fontsize=16
)

plt.tight_layout()

combined_path = (
    OUTPUT_DIR /
    "dataset_visual_inspection.png"
)

plt.savefig(
    combined_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print()
print(
    f"Combined visualization saved to:"
)
print(
    combined_path.resolve()
)

print()
print("=" * 70)
print("VISUAL INSPECTION COMPLETE")
print("=" * 70)