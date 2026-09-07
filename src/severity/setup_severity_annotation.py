from pathlib import Path
import shutil
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SELECTION_CSV = PROJECT_ROOT / "outputs" / "severity_selection.csv"
DATASET_ROOT = PROJECT_ROOT / "dataset"

SEVERITY_ROOT = PROJECT_ROOT / "outputs" / "severity"
ANNOTATION_IMAGES = SEVERITY_ROOT / "annotation_images"
PILOT_IMAGES = SEVERITY_ROOT / "pilot" / "images"

# Copy only rows marked low/moderate/high.
VALID_SELECTIONS = {"low", "moderate", "high"}

CLASS_FOLDERS = {
    "apple_scab": "Apple___Apple_scab",
    "black_rot": "Apple___Black_rot",
    "cedar_rust": "Apple___Cedar_apple_rust",
    "healthy": "Apple___healthy",
}


def main():
    if not SELECTION_CSV.exists():
        raise FileNotFoundError(f"Selection CSV not found: {SELECTION_CSV}")

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_ROOT}\n"
            "Run this script from your project where the dataset/ folder exists."
        )

    df = pd.read_csv(SELECTION_CSV)

    required = {"image_path", "internal_class", "selection", "reviewed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    # Only reviewed and non-excluded images are sent to annotation.
    df = df[
        (df["reviewed"].astype(str) == "1")
        & (df["selection"].astype(str).str.lower().isin(VALID_SELECTIONS))
    ].copy()

    # Recreate annotation folder structure without deleting existing annotations.
    for class_name in CLASS_FOLDERS:
        (ANNOTATION_IMAGES / class_name).mkdir(parents=True, exist_ok=True)

    copied = []
    missing_files = []

    for _, row in df.iterrows():
        rel_path = Path(str(row["image_path"]).replace("\\", "/"))
        source = PROJECT_ROOT / rel_path

        class_name = str(row["internal_class"]).strip()
        if class_name not in CLASS_FOLDERS:
            # Fall back to the class column if needed.
            class_name = str(row.get("class", "")).strip().lower()
            aliases = {
                "apple scab": "apple_scab",
                "black rot": "black_rot",
                "cedar apple rust": "cedar_rust",
                "healthy": "healthy",
            }
            class_name = aliases.get(class_name, class_name)

        if class_name not in CLASS_FOLDERS:
            missing_files.append((str(row["image_path"]), "unknown_class"))
            continue

        destination = ANNOTATION_IMAGES / class_name / source.name

        if not source.exists():
            missing_files.append((str(row["image_path"]), "file_not_found"))
            continue

        shutil.copy2(source, destination)

        copied.append({
            "image_path": str(row["image_path"]),
            "class": str(row.get("class", "")),
            "internal_class": class_name,
            "selection": str(row["selection"]).lower(),
            "reviewed": int(row["reviewed"]),
            "annotation_image": str(destination.relative_to(PROJECT_ROOT)),
        })

    manifest = pd.DataFrame(copied)
    manifest_path = SEVERITY_ROOT / "annotation_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    missing_path = SEVERITY_ROOT / "missing_annotation_files.csv"
    pd.DataFrame(
        missing_files, columns=["image_path", "reason"]
    ).to_csv(missing_path, index=False)

    # Print summary.
    print("=" * 70)
    print("SEVERITY ANNOTATION IMAGE SETUP")
    print("=" * 70)
    print(f"Selection CSV : {SELECTION_CSV}")
    print(f"Dataset root  : {DATASET_ROOT}")
    print(f"Output folder : {ANNOTATION_IMAGES}")
    print()
    print(f"Selected rows : {len(df)}")
    print(f"Copied images : {len(copied)}")
    print(f"Missing files : {len(missing_files)}")
    print()
    if not manifest.empty:
        print("Copied by class:")
        print(manifest["internal_class"].value_counts().to_string())
        print()
        print("Copied by provisional selection:")
        print(manifest["selection"].value_counts().to_string())
    print()
    print(f"Manifest saved: {manifest_path}")
    print(f"Missing list : {missing_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
