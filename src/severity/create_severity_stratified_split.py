"""
Create a severity-stratified train/validation/test split.

Locked severity grades:
    Grade 0: 0-5%
    Grade 1: >5-15%
    Grade 2: >15-30%
    Grade 3: >30-100%

The split is stratified by severity grade so every grade is represented
in train/validation/test as far as the available samples allow.

Disease class distribution is reported after splitting but is NOT used
as the primary stratification variable because some class+grade
combinations are too rare (e.g. Black Rot Grade 3).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GROUND_TRUTH_CSV = (
    PROJECT_ROOT / "outputs" / "severity" / "severity_ground_truth.csv"
)
OUTPUT_CSV = (
    PROJECT_ROOT / "outputs" / "severity" / "severity_stratified_split.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # fraction of the complete dataset


def severity_grade(si):
    if si < 0:
        raise ValueError(f"Negative severity index: {si}")
    if si <= 5:
        return 0
    if si <= 15:
        return 1
    if si <= 30:
        return 2
    return 3


def infer_class(mask_path):
    parts = str(mask_path).replace("\\", "/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Cannot infer class from mask_path: {mask_path}")
    return parts[-2]


def main():
    if not GROUND_TRUTH_CSV.exists():
        raise FileNotFoundError(GROUND_TRUTH_CSV)

    gt = pd.read_csv(GROUND_TRUTH_CSV)

    required = {"mask_path", "severity_index_percent"}
    missing = required - set(gt.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    gt = gt.copy()
    gt["class"] = (
        gt["class"].astype(str)
        if "class" in gt.columns
        else gt["mask_path"].map(infer_class)
    )
    gt["filename"] = gt["mask_path"].map(
        lambda x: Path(str(x).replace("\\", "/")).name
    )
    gt["severity_grade"] = gt["severity_index_percent"].map(severity_grade)

    if gt["severity_grade"].value_counts().min() < 3:
        raise ValueError(
            "A severity grade has fewer than 3 samples; "
            "a train/validation/test split cannot be made reliably."
        )

    # First: isolate the final test set.
    train_val, test = train_test_split(
        gt,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=gt["severity_grade"],
    )

    # Validation is 15% of the complete dataset.
    # Convert that to a fraction of the remaining 85%.
    val_fraction_of_train_val = VAL_SIZE / (1.0 - TEST_SIZE)

    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_train_val,
        random_state=RANDOM_STATE,
        stratify=train_val["severity_grade"],
    )

    train = train.copy()
    val = val.copy()
    test = test.copy()

    train["split"] = "train"
    val["split"] = "validation"
    test["split"] = "test"

    result = pd.concat([train, val, test], ignore_index=True)

    # Keep a stable, useful column order.
    columns = [
        "mask_path",
        "class",
        "filename",
        "severity_index_percent",
        "severity_grade",
        "split",
    ]
    result = result[columns].sort_values(
        ["split", "severity_grade", "class", "filename"]
    ).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)

    print("=" * 70)
    print("SEVERITY-STRATIFIED SPLIT")
    print("=" * 70)
    print(f"Source: {GROUND_TRUTH_CSV}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Total: {len(result)}")
    print()

    print("Overall grade distribution:")
    overall = (
        result.groupby("severity_grade")
        .size()
        .reindex([0, 1, 2, 3], fill_value=0)
    )
    for grade, count in overall.items():
        print(f"  Grade {grade}: {count:3d} ({100*count/len(result):5.2f}%)")

    print("\nSplit sizes:")
    print(result["split"].value_counts().reindex(
        ["train", "validation", "test"], fill_value=0
    ).to_string())

    print("\nGrade distribution by split:")
    table = pd.crosstab(result["split"], result["severity_grade"])
    table = table.reindex(
        index=["train", "validation", "test"],
        columns=[0, 1, 2, 3],
        fill_value=0,
    )
    table.columns = ["grade_0", "grade_1", "grade_2", "grade_3"]
    print(table.to_string())

    print("\nDisease distribution by split:")
    disease_table = pd.crosstab(result["split"], result["class"])
    disease_table = disease_table.reindex(
        index=["train", "validation", "test"], fill_value=0
    )
    print(disease_table.to_string())

    print("\nGrade percentages within each split:")
    pct = table.div(table.sum(axis=1), axis=0) * 100
    print(pct.round(2).to_string())

    print("\nSaved:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
