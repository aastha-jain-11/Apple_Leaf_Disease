from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

METADATA_PATH = Path("outputs/metadata.csv")
OUTPUT_PATH = Path("outputs/dataset_split.csv")

RANDOM_SEED = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# CHECK CONFIGURATION
# ============================================================

assert abs(
    TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE - 1.0
) < 1e-6, "Split percentages must add up to 1."


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 70)
print("CREATING DATASET SPLIT")
print("=" * 70)

print()

print(f"Reading metadata from:")
print(METADATA_PATH.resolve())

df = pd.read_csv(METADATA_PATH)

print()
print(f"Total rows in metadata: {len(df)}")


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

if "duplicate" in df.columns:

    duplicate_count = int(
        df["duplicate"].fillna(False).astype(bool).sum()
    )

    print(f"Duplicate rows found: {duplicate_count}")

    df = df[
        ~df["duplicate"].fillna(False).astype(bool)
    ].copy()

else:

    print("No duplicate column found.")

print(f"Rows after duplicate removal: {len(df)}")


# ============================================================
# REMOVE CORRUPTED IMAGES
# ============================================================

# The analyze_dataset.py script only places readable images
# into metadata.csv, so normally corrupted images are already
# excluded.

print()
print("Checking image paths...")

existing_mask = df["image_path"].apply(
    lambda path: Path(path).exists()
)

missing_count = int((~existing_mask).sum())

if missing_count > 0:

    print(
        f"WARNING: {missing_count} image paths "
        f"do not exist."
    )

    df = df[existing_mask].copy()

else:

    print("All image paths exist.")


# ============================================================
# STANDARDIZE CLASS NAMES
# ============================================================

LABEL_MAP = {

    # PlantVillage names
    "Apple___Apple_scab": "apple_scab",
    "Apple___Black_rot": "black_rot",
    "Apple___Cedar_apple_rust": "cedar_rust",
    "Apple___healthy": "healthy",

    # Common alternative names
    "Apple Scab": "apple_scab",
    "Apple Scab": "apple_scab",

    "Black Rot": "black_rot",
    "Apple Black Rot": "black_rot",

    "Cedar Apple Rust": "cedar_rust",
    "Cedar Rust": "cedar_rust",
    "Apple Cedar Rust": "cedar_rust",

    "Healthy": "healthy",
    "Apple Healthy": "healthy",
}


# Check which column contains the class name
if "class" in df.columns:

    source_label_column = "class"

elif "label" in df.columns:

    source_label_column = "label"

else:

    raise ValueError(
        "Could not find a class/label column in metadata.csv"
    )


df["label"] = df[source_label_column].map(LABEL_MAP)


# ============================================================
# CHECK FOR UNKNOWN LABELS
# ============================================================

unknown_labels = df[
    df["label"].isna()
][source_label_column].unique()


if len(unknown_labels) > 0:

    print()
    print("ERROR: Unknown class names found:")
    print()

    for label in unknown_labels:
        print(f"  {label}")

    print()
    print(
        "Add these names to LABEL_MAP and run the "
        "script again."
    )

    raise SystemExit(1)


# ============================================================
# SHOW CLEAN CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("CLEAN CLASS DISTRIBUTION")
print("=" * 70)

print(
    df["label"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# FIRST SPLIT
#
# 70% train
# 30% temporary
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=(VALIDATION_SIZE + TEST_SIZE),
    random_state=RANDOM_SEED,
    stratify=df["label"]
)


# ============================================================
# SECOND SPLIT
#
# temp = 30%
#
# We need:
# validation = 15%
# test       = 15%
#
# Therefore:
# validation = 50% of temp
# test       = 50% of temp
# ============================================================

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=RANDOM_SEED,
    stratify=temp_df["label"]
)


# ============================================================
# ADD SPLIT COLUMN
# ============================================================

train_df = train_df.copy()
validation_df = validation_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
validation_df["split"] = "validation"
test_df["split"] = "test"


# ============================================================
# COMBINE
# ============================================================

final_df = pd.concat(
    [
        train_df,
        validation_df,
        test_df
    ],
    ignore_index=True
)


# ============================================================
# SHUFFLE FINAL CSV
# ============================================================

final_df = final_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

final_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY SPLIT COUNTS
# ============================================================

print()
print("=" * 70)
print("FINAL SPLIT")
print("=" * 70)

print()

print(
    final_df["split"]
    .value_counts()
    .to_string()
)

print()

print("Class distribution by split:")
print()

distribution = pd.crosstab(
    final_df["label"],
    final_df["split"]
)

print(distribution.to_string())


# ============================================================
# CHECK FOR DATA LEAKAGE
# ============================================================

print()
print("=" * 70)
print("DATA LEAKAGE CHECK")
print("=" * 70)

# Each image path should occur only once.
duplicate_paths = (
    final_df["image_path"]
    .duplicated()
    .sum()
)

print(
    f"Repeated image paths across splits: "
    f"{duplicate_paths}"
)

if duplicate_paths == 0:

    print("PASS: No repeated image paths.")

else:

    print(
        "WARNING: Repeated image paths detected!"
    )


# ============================================================
# FINAL INFORMATION
# ============================================================

print()
print("=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

print()

print("Output file:")
print(OUTPUT_PATH.resolve())

print()

print("Random seed:")
print(RANDOM_SEED)