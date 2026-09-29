"""
Four-grade severity evaluation.

Uses the R1 Random Forest regression model:
    lesion_area_percent + lesion_count

Ground-truth grades are derived from manually annotated severity index:
    0-20     -> Grade 0
    >20-40   -> Grade 1
    >40-60   -> Grade 2
    >60-100  -> Grade 3

The test predictions saved by train_severity_regression.py are evaluated.
This script does not retrain or alter SegFormer.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "severity"
    / "regression"
    / "R1_area_plus_count_test_predictions.csv"
)
OUT_DIR = PROJECT_ROOT / "outputs" / "severity" / "grades"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BINS = [0.0, 20.0, 40.0, 60.0, 100.000001]
LABELS = [0, 1, 2, 3]


def si_to_grade(si):
    """Map severity index (%) to the four project severity grades."""
    if si <= 20:
        return 0
    if si <= 40:
        return 1
    if si <= 60:
        return 2
    return 3


def main():
    print("=" * 70)
    print("FOUR-GRADE SEVERITY EVALUATION")
    print("=" * 70)

    if not PRED_CSV.exists():
        raise FileNotFoundError(
            f"Missing R1 predictions: {PRED_CSV}\n"
            "Run train_severity_regression.py first."
        )

    df = pd.read_csv(PRED_CSV)

    required = {"ground_truth_si", "predicted_si", "filename", "class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["ground_truth_grade"] = df["ground_truth_si"].apply(si_to_grade)

    # Keep predicted SI in the physically valid range before grade mapping.
    df["predicted_si_clipped"] = df["predicted_si"].clip(0, 100)
    df["predicted_grade"] = df["predicted_si_clipped"].apply(si_to_grade)

    y_true = df["ground_truth_grade"].to_numpy()
    y_pred = df["predicted_grade"].to_numpy()

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
    )
    kappa = cohen_kappa_score(y_true, y_pred, labels=LABELS)
    grade_mae = mean_absolute_error(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cohen_kappa": kappa,
        "grade_mae": grade_mae,
        "n_test": len(df),
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(OUT_DIR / "severity_grade_metrics.csv", index=False)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_grade_{x}" for x in LABELS],
        columns=[f"pred_grade_{x}" for x in LABELS],
    )
    cm_df.to_csv(OUT_DIR / "severity_grade_confusion_matrix.csv")

    pred_out = df[
        [
            "filename",
            "class",
            "ground_truth_si",
            "predicted_si",
            "predicted_si_clipped",
            "ground_truth_grade",
            "predicted_grade",
        ]
    ].copy()
    pred_out.to_csv(OUT_DIR / "severity_grade_predictions.csv", index=False)

    print("\nGRADE THRESHOLDS")
    print("Grade 0: 0-20%")
    print("Grade 1: >20-40%")
    print("Grade 2: >40-60%")
    print("Grade 3: >60-100%")

    print("\nTEST RESULTS")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")
    print(f"Cohen Kappa:  {kappa:.4f}")
    print(f"Grade MAE:    {grade_mae:.4f}")

    print("\nCONFUSION MATRIX")
    print(cm_df.to_string())

    print("\nSaved:")
    print(OUT_DIR / "severity_grade_metrics.csv")
    print(OUT_DIR / "severity_grade_confusion_matrix.csv")
    print(OUT_DIR / "severity_grade_predictions.csv")


if __name__ == "__main__":
    main()
