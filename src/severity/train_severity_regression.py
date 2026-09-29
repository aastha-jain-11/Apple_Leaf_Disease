"""
Severity regression experiments.

Target:
    Manual severity index calculated from the LabelMe ground-truth masks.

Inputs:
    Features extracted from SegFormer-predicted masks.

Experiments:
    R0: predicted lesion area % only (direct baseline)
    R1: area % + lesion count
    R2: area % + lesion count + lesion-size features

Evaluation:
    Held-out TEST split only.
    Metrics: MAE, RMSE, R2.

The split is inherited from severity_segmentation_split.csv, so no new
random split is created here.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_CSV = PROJECT_ROOT / "outputs" / "severity" / "severity_features_predicted.csv"
SPLIT_CSV = PROJECT_ROOT / "outputs" / "severity" / "severity_segmentation_split.csv"
GROUND_TRUTH_DIR = PROJECT_ROOT / "outputs" / "severity"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "severity" / "regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def find_ground_truth_csv():
    candidates = sorted(
        GROUND_TRUTH_DIR.glob("severity_ground_truth*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No severity_ground_truth*.csv found in outputs/severity/"
        )
    return candidates[0]


def normalize_filename(name):
    """
    Match filenames robustly despite .png/.JPG/.jpeg extension differences.
    """
    return Path(str(name)).stem.lower().strip()


def load_ground_truth():
    gt_path = find_ground_truth_csv()
    gt = pd.read_csv(gt_path)

    required = {"mask_path", "severity_index_percent"}
    missing = required - set(gt.columns)
    if missing:
        raise ValueError(
            f"Ground-truth CSV is missing columns: {sorted(missing)}"
        )

    # mask_path examples:
    # masks/apple_scab/image (115).png
    gt["filename_key"] = gt["mask_path"].map(normalize_filename)

    # Class is encoded in mask_path when not explicitly present.
    if "class" not in gt.columns:
        gt["class"] = gt["mask_path"].astype(str).map(
            lambda x: Path(x.replace("\\", "/")).parts[-2]
        )

    gt["class"] = gt["class"].astype(str)
    gt["filename_key"] = gt["filename_key"].astype(str)

    # There should be one GT value per image.
    gt = gt.drop_duplicates(subset=["class", "filename_key"])

    print(f"Ground truth: {gt_path}")
    return gt


def prepare_data():
    features = pd.read_csv(FEATURE_CSV)
    split = pd.read_csv(SPLIT_CSV)
    gt = load_ground_truth()

    features["filename_key"] = features["filename"].map(normalize_filename)
    split["filename_key"] = split["filename"].map(normalize_filename)

    # Use the split CSV as the authoritative split assignment.
    data = features.merge(
        split[["class", "filename_key", "split"]],
        on=["class", "filename_key"],
        how="inner",
        suffixes=("", "_split"),
    )

    data = data.merge(
        gt[["class", "filename_key", "severity_index_percent"]],
        on=["class", "filename_key"],
        how="inner",
    )

    data = data.rename(
        columns={"severity_index_percent": "ground_truth_si"}
    )

    if data.empty:
        raise RuntimeError(
            "No rows matched between predicted features, split CSV, and ground truth."
        )

    if data["ground_truth_si"].isna().any():
        raise RuntimeError("Ground-truth SI contains missing values.")

    return data


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def train_rf(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def main():
    print("=" * 70)
    print("SEVERITY REGRESSION EXPERIMENTS")
    print("=" * 70)

    if not FEATURE_CSV.exists():
        raise FileNotFoundError(FEATURE_CSV)
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(SPLIT_CSV)

    data = prepare_data()

    train = data[data["split"].str.lower() == "train"].copy()
    val = data[data["split"].str.lower().isin(["val", "validation"])].copy()
    test = data[data["split"].str.lower() == "test"].copy()

    print(f"Matched images: {len(data)}")
    print(f"Train: {len(train)} | Validation: {len(val)} | Test: {len(test)}")
    print(
        f"Ground-truth SI range: "
        f"{data['ground_truth_si'].min():.2f}% - "
        f"{data['ground_truth_si'].max():.2f}%"
    )

    feature_sets = {
        "R0_area_only": [
            "lesion_area_percent",
        ],
        "R1_area_plus_count": [
            "lesion_area_percent",
            "lesion_count",
        ],
        "R2_area_count_size": [
            "lesion_area_percent",
            "lesion_count",
            "mean_lesion_area",
            "max_lesion_area",
            "lesion_area_std",
        ],
    }

    results = []

    # Direct area baseline: no learned regression.
    y_test = test["ground_truth_si"].to_numpy()
    direct_pred = test["lesion_area_percent"].to_numpy()
    metrics = evaluate(y_test, direct_pred)
    results.append(
        {
            "model": "R0_area_only_direct",
            "features": "lesion_area_percent",
            **metrics,
        }
    )

    # Learned models. Validation set is used only for reporting, not for
    # test selection; the RF configuration is fixed in this script.
    for name in ["R1_area_plus_count", "R2_area_count_size"]:
        cols = feature_sets[name]

        X_train = train[cols].fillna(0).to_numpy()
        y_train = train["ground_truth_si"].to_numpy()

        X_val = val[cols].fillna(0).to_numpy()
        y_val = val["ground_truth_si"].to_numpy()

        X_test = test[cols].fillna(0).to_numpy()

        model = train_rf(X_train, y_train)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_metrics = evaluate(y_val, val_pred)
        test_metrics = evaluate(y_test, test_pred)

        results.append(
            {
                "model": name,
                "features": ", ".join(cols),
                "val_MAE": val_metrics["MAE"],
                "val_RMSE": val_metrics["RMSE"],
                "val_R2": val_metrics["R2"],
                "MAE": test_metrics["MAE"],
                "RMSE": test_metrics["RMSE"],
                "R2": test_metrics["R2"],
            }
        )

        # Save test predictions for later grade analysis.
        pred_df = test[
            ["filename", "class", "ground_truth_si", "lesion_area_percent", "lesion_count"]
        ].copy()
        pred_df["predicted_si"] = test_pred
        pred_df.to_csv(
            OUTPUT_DIR / f"{name}_test_predictions.csv",
            index=False,
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "severity_regression_results.csv", index=False)

    print("\nTEST RESULTS")
    print(results_df.to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_DIR / "severity_regression_results.csv")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
