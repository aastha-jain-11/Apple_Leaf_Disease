"""
Evaluate four severity grades from the R1 continuous SI predictions.

Locked grading thresholds:
    Grade 0: 0-5%
    Grade 1: >5-15%
    Grade 2: >15-30%
    Grade 3: >30-100%

Important:
    The continuous SI regression remains the primary severity prediction.
    Grades are a derived ordinal representation.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "severity"
    / "regression"
    / "R1_area_plus_count_stratified_test_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "severity" / "grades"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRADE_LABELS = [0, 1, 2, 3]


def severity_grade(si):
    if si < 0:
        raise ValueError(f"Negative SI: {si}")
    if si <= 5:
        return 0
    if si <= 15:
        return 1
    if si <= 30:
        return 2
    return 3


def main():
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(PREDICTIONS_CSV)

    df = pd.read_csv(PREDICTIONS_CSV)

    required = {
        "class",
        "filename",
        "ground_truth_si",
        "predicted_si",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["true_grade"] = df["ground_truth_si"].map(severity_grade)
    df["pred_grade"] = df["predicted_si"].clip(0, 100).map(
        severity_grade
    )

    y_true = df["true_grade"]
    y_pred = df["pred_grade"]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=GRADE_LABELS,
    )

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, labels=GRADE_LABELS,
        average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_true, y_pred, labels=GRADE_LABELS,
        average="weighted", zero_division=0
    )
    kappa = cohen_kappa_score(y_true, y_pred, labels=GRADE_LABELS)
    grade_mae = mean_absolute_error(y_true, y_pred)

    print("=" * 70)
    print("FOUR-GRADE SEVERITY EVALUATION")
    print("=" * 70)

    print("\nGRADE THRESHOLDS")
    print("Grade 0: 0-5%")
    print("Grade 1: >5-15%")
    print("Grade 2: >15-30%")
    print("Grade 3: >30-100%")

    print("\nTEST SIZE:", len(df))

    print("\nTRUE GRADE DISTRIBUTION")
    print(
        y_true.value_counts()
        .reindex(GRADE_LABELS, fill_value=0)
        .to_string()
    )

    print("\nPREDICTED GRADE DISTRIBUTION")
    print(
        y_pred.value_counts()
        .reindex(GRADE_LABELS, fill_value=0)
        .to_string()
    )

    print("\nTEST RESULTS")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")
    print(f"Cohen Kappa:  {kappa:.4f}")
    print(f"Grade MAE:    {grade_mae:.4f}")

    print("\nCONFUSION MATRIX")
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_grade_{g}" for g in GRADE_LABELS],
        columns=[f"pred_grade_{g}" for g in GRADE_LABELS],
    )
    print(cm_df.to_string())

    print("\nCLASSIFICATION REPORT")
    report = classification_report(
        y_true,
        y_pred,
        labels=GRADE_LABELS,
        target_names=[
            "Grade 0",
            "Grade 1",
            "Grade 2",
            "Grade 3",
        ],
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose()
    print(report_df.to_string())

    metrics = pd.DataFrame([{
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cohen_kappa": kappa,
        "grade_mae": grade_mae,
        "test_samples": len(df),
    }])

    metrics_path = OUTPUT_DIR / "severity_grade_metrics_stratified.csv"
    cm_path = OUTPUT_DIR / "severity_grade_confusion_matrix_stratified.csv"
    predictions_path = OUTPUT_DIR / "severity_grade_predictions_stratified.csv"
    report_path = OUTPUT_DIR / "severity_grade_classification_report_stratified.csv"

    metrics.to_csv(metrics_path, index=False)
    cm_df.to_csv(cm_path)
    df.to_csv(predictions_path, index=False)
    report_df.to_csv(report_path)

    print("\nSaved:")
    print(metrics_path)
    print(cm_path)
    print(predictions_path)
    print(report_path)


if __name__ == "__main__":
    main()
