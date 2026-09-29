"""
Severity regression using the locked severity-stratified split.

Target:
    Manual continuous Severity Index (SI), 0-100%.

Inputs:
    SegFormer-predicted severity features.

Experiments:
    R0: predicted lesion area percentage only (direct baseline)
    R1: area percentage + lesion count
    R2: area percentage + lesion count + lesion-size features

The new split is stratified by the four locked severity grades:
    0: 0-5%
    1: >5-15%
    2: >15-30%
    3: >30-100%

The model is trained only on the training split.
Validation is reported separately.
The test set is held out for final evaluation.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_CSV = (
    PROJECT_ROOT / "outputs" / "severity" / "severity_features_predicted.csv"
)
SPLIT_CSV = (
    PROJECT_ROOT / "outputs" / "severity" / "severity_stratified_split.csv"
)
GROUND_TRUTH_CSV = (
    PROJECT_ROOT / "outputs" / "severity" / "severity_ground_truth.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "severity" / "regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

FEATURES = {
    "R0_area_only": ["lesion_area_percent"],
    "R1_area_plus_count": ["lesion_area_percent", "lesion_count"],
    "R2_area_count_size": [
        "lesion_area_percent",
        "lesion_count",
        "mean_lesion_area",
        "max_lesion_area",
        "lesion_area_std",
    ],
}


def normalize_filename(name):
    return Path(str(name)).stem.lower().strip()


def load_data():
    features = pd.read_csv(FEATURE_CSV)
    split = pd.read_csv(SPLIT_CSV)
    gt = pd.read_csv(GROUND_TRUTH_CSV)

    features["filename_key"] = features["filename"].map(normalize_filename)
    split["filename_key"] = split["filename"].map(normalize_filename)
    gt["filename_key"] = gt["mask_path"].map(normalize_filename)

    if "class" not in gt.columns:
        gt["class"] = gt["mask_path"].astype(str).map(
            lambda x: Path(x.replace("\\", "/")).parts[-2]
        )

    # The predicted-feature CSV may already contain an old `split` column.
    # Drop it before merging so the NEW severity-stratified split remains
    # the authoritative split assignment.
    features = features.drop(
        columns=["split", "severity_grade"],
        errors="ignore",
    )

    data = features.merge(
        split[["class", "filename_key", "split", "severity_grade"]],
        on=["class", "filename_key"],
        how="inner",
    )

    data = data.merge(
        gt[["class", "filename_key", "severity_index_percent"]],
        on=["class", "filename_key"],
        how="inner",
    )

    data = data.rename(
        columns={"severity_index_percent": "ground_truth_si"}
    )

    if len(data) != 693:
        print(f"WARNING: matched rows = {len(data)}, expected 693.")

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
    print("SEVERITY REGRESSION - STRATIFIED SPLIT")
    print("=" * 70)

    for path in [FEATURE_CSV, SPLIT_CSV, GROUND_TRUTH_CSV]:
        if not path.exists():
            raise FileNotFoundError(path)

    data = load_data()

    train = data[data["split"] == "train"].copy()
    val = data[data["split"] == "validation"].copy()
    test = data[data["split"] == "test"].copy()

    print(f"Matched images: {len(data)}")
    print(
        f"Train: {len(train)} | "
        f"Validation: {len(val)} | "
        f"Test: {len(test)}"
    )

    print("\nTest grade distribution:")
    print(
        test["severity_grade"]
        .value_counts()
        .reindex([0, 1, 2, 3], fill_value=0)
        .to_string()
    )

    results = []
    prediction_tables = {}

    for name, feature_cols in FEATURES.items():
        print("\n" + "-" * 70)
        print(name)
        print("Features:", ", ".join(feature_cols))

        X_train = train[feature_cols]
        y_train = train["ground_truth_si"]

        X_val = val[feature_cols]
        y_val = val["ground_truth_si"]

        X_test = test[feature_cols]
        y_test = test["ground_truth_si"]

        if name == "R0_area_only":
            # Direct baseline: no learned regression.
            val_pred = X_val["lesion_area_percent"].to_numpy()
            test_pred = X_test["lesion_area_percent"].to_numpy()
            train_pred = X_train["lesion_area_percent"].to_numpy()
        else:
            model = train_rf(X_train, y_train)
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)

        train_metrics = evaluate(y_train, train_pred)
        val_metrics = evaluate(y_val, val_pred)
        test_metrics = evaluate(y_test, test_pred)

        print(
            f"Train: MAE={train_metrics['MAE']:.4f}, "
            f"RMSE={train_metrics['RMSE']:.4f}, "
            f"R2={train_metrics['R2']:.4f}"
        )
        print(
            f"Validation: MAE={val_metrics['MAE']:.4f}, "
            f"RMSE={val_metrics['RMSE']:.4f}, "
            f"R2={val_metrics['R2']:.4f}"
        )
        print(
            f"Test: MAE={test_metrics['MAE']:.4f}, "
            f"RMSE={test_metrics['RMSE']:.4f}, "
            f"R2={test_metrics['R2']:.4f}"
        )

        results.append({
            "model": name,
            "features": ", ".join(feature_cols),
            "train_MAE": train_metrics["MAE"],
            "train_RMSE": train_metrics["RMSE"],
            "train_R2": train_metrics["R2"],
            "validation_MAE": val_metrics["MAE"],
            "validation_RMSE": val_metrics["RMSE"],
            "validation_R2": val_metrics["R2"],
            "test_MAE": test_metrics["MAE"],
            "test_RMSE": test_metrics["RMSE"],
            "test_R2": test_metrics["R2"],
        })

        prediction_tables[name] = pd.DataFrame({
            "class": test["class"].to_numpy(),
            "filename": test["filename"].to_numpy(),
            "severity_grade_true": test["severity_grade"].to_numpy(),
            "ground_truth_si": y_test.to_numpy(),
            "predicted_si": test_pred,
        })

    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / "severity_regression_stratified_results.csv"
    results_df.to_csv(results_path, index=False)

    # Save R1 test predictions because R1 is the selected model from the
    # previous experiment and will feed the final grade evaluation.
    r1_path = (
        OUTPUT_DIR
        / "R1_area_plus_count_stratified_test_predictions.csv"
    )
    prediction_tables["R1_area_plus_count"].to_csv(r1_path, index=False)

    print("\n" + "=" * 70)
    print("SAVED")
    print("=" * 70)
    print(results_path)
    print(r1_path)


if __name__ == "__main__":
    main()
