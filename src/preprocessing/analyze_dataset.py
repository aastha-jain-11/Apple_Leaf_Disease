from pathlib import Path
from PIL import Image
import pandas as pd
import hashlib


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path("dataset")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# STORAGE
# ============================================================

records = []
bad_files = []
hashes = {}


# ============================================================
# SCAN DATASET
# ============================================================

print("Scanning dataset...")
print(f"Dataset location: {DATASET_ROOT.resolve()}")
print()


for file_path in DATASET_ROOT.rglob("*"):

    # Ignore folders and unsupported files
    if not file_path.is_file():
        continue

    if file_path.suffix.lower() not in VALID_EXTENSIONS:
        continue

    try:

        # ----------------------------------------------------
        # Check whether image is readable
        # ----------------------------------------------------

        with Image.open(file_path) as image:

            image.verify()

        # Re-open because verify() closes/invalidates the image
        with Image.open(file_path) as image:

            width, height = image.size
            image_format = image.format
            mode = image.mode


        # ----------------------------------------------------
        # Calculate file hash
        # ----------------------------------------------------

        file_hash = hashlib.md5(
            file_path.read_bytes()
        ).hexdigest()


        # ----------------------------------------------------
        # Check exact duplicate
        # ----------------------------------------------------

        is_duplicate = file_hash in hashes

        if not is_duplicate:
            hashes[file_hash] = str(file_path)


        # ----------------------------------------------------
        # Class name
        #
        # Your current directory structure is:
        #
        # dataset/
        #     Apple___Apple_scab/
        #     Apple___Black_rot/
        #     ...
        # ----------------------------------------------------

        class_name = file_path.parent.name


        # ----------------------------------------------------
        # Save information
        # ----------------------------------------------------

        records.append({
            "image_path": str(file_path),
            "class": class_name,
            "width": width,
            "height": height,
            "format": image_format,
            "mode": mode,
            "duplicate": is_duplicate
        })


    except Exception as error:

        bad_files.append({
            "image_path": str(file_path),
            "error": str(error)
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)


# ============================================================
# SAVE METADATA
# ============================================================

metadata_path = OUTPUT_DIR / "metadata.csv"

df.to_csv(
    metadata_path,
    index=False
)


# ============================================================
# SAVE BAD IMAGE LIST
# ============================================================

if bad_files:

    bad_images_path = OUTPUT_DIR / "bad_images.csv"

    pd.DataFrame(bad_files).to_csv(
        bad_images_path,
        index=False
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 60)
print("DATASET ANALYSIS")
print("=" * 60)

print()

print(f"Total valid images      : {len(df)}")
print(f"Corrupted/unreadable    : {len(bad_files)}")

if len(df) > 0:
    print(
        f"Exact duplicate images : "
        f"{df['duplicate'].sum()}"
    )

print()


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("Images per class")
print("-" * 60)

if len(df) > 0:

    print(
        df["class"]
        .value_counts()
        .to_string()
    )

print()


# ============================================================
# IMAGE FORMATS
# ============================================================

print("Image formats")
print("-" * 60)

if len(df) > 0:

    print(
        df["format"]
        .value_counts()
        .to_string()
    )

print()


# ============================================================
# IMAGE MODES
# ============================================================

print("Image modes")
print("-" * 60)

if len(df) > 0:

    print(
        df["mode"]
        .value_counts()
        .to_string()
    )

print()


# ============================================================
# IMAGE DIMENSIONS
# ============================================================

print("Image dimensions")
print("-" * 60)

if len(df) > 0:

    print(
        df[["width", "height"]]
        .describe()
        .to_string()
    )

print()


# ============================================================
# SAVE LOCATION
# ============================================================

print("=" * 60)
print("Analysis complete.")
print("=" * 60)

print()

print(f"Metadata saved to:")
print(metadata_path.resolve())

if bad_files:

    print()
    print("Bad image list saved to:")
    print(
        (OUTPUT_DIR / "bad_images.csv").resolve()
    )