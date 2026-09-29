from pathlib import Path
import json
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = PROJECT_ROOT / "outputs" / "severity" / "annotation_images"
OUT_ROOT = PROJECT_ROOT / "outputs" / "severity"
REPORT = OUT_ROOT / "annotation_qa_report.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
EXPECTED_CLASSES = {"apple_scab", "black_rot", "cedar_rust", "healthy"}
EXPECTED_LABELS = {"leaf", "lesion"}

def main():
    rows = []

    for class_dir in sorted(IMAGE_ROOT.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        if class_name not in EXPECTED_CLASSES:
            continue

        images = [p for p in class_dir.iterdir() if p.suffix in IMAGE_EXTS]

        for image_path in images:
            json_path = image_path.with_suffix(".json")

            row = {
                "class": class_name,
                "image": str(image_path.relative_to(PROJECT_ROOT)),
                "json": str(json_path.relative_to(PROJECT_ROOT)),
                "json_exists": json_path.exists(),
                "image_width": None,
                "image_height": None,
                "json_width": None,
                "json_height": None,
                "labels": "",
                "leaf_count": 0,
                "lesion_count": 0,
                "unknown_label_count": 0,
                "invalid_polygon_count": 0,
                "healthy_has_lesion": False,
                "disease_has_no_lesion": False,
                "status": "OK",
                "issues": "",
            }

            issues = []

            try:
                with Image.open(image_path) as im:
                    row["image_width"], row["image_height"] = im.size
            except Exception as e:
                issues.append(f"image_open_error:{e}")

            if not json_path.exists():
                issues.append("missing_json")
                row["status"] = "FAIL"
                row["issues"] = ";".join(issues)
                rows.append(row)
                continue

            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                issues.append(f"invalid_json:{e}")
                row["status"] = "FAIL"
                row["issues"] = ";".join(issues)
                rows.append(row)
                continue

            row["json_width"] = data.get("imageWidth")
            row["json_height"] = data.get("imageHeight")

            if (
                row["image_width"] is not None
                and row["json_width"] is not None
                and row["image_width"] != row["json_width"]
            ):
                issues.append("width_mismatch")

            if (
                row["image_height"] is not None
                and row["json_height"] is not None
                and row["image_height"] != row["json_height"]
            ):
                issues.append("height_mismatch")

            labels = []
            for shape in data.get("shapes", []):
                label = str(shape.get("label", "")).strip().lower()
                labels.append(label)

                if label == "leaf":
                    row["leaf_count"] += 1
                elif label == "lesion":
                    row["lesion_count"] += 1
                elif label:
                    row["unknown_label_count"] += 1

                points = shape.get("points", [])
                if shape.get("shape_type") == "polygon" and len(points) < 3:
                    row["invalid_polygon_count"] += 1

            row["labels"] = ",".join(sorted(set(labels)))

            if row["leaf_count"] == 0:
                issues.append("missing_leaf")

            if row["unknown_label_count"] > 0:
                issues.append("unknown_label")

            if row["invalid_polygon_count"] > 0:
                issues.append("invalid_polygon")

            if class_name == "healthy" and row["lesion_count"] > 0:
                row["healthy_has_lesion"] = True
                issues.append("healthy_has_lesion")

            if class_name != "healthy" and row["lesion_count"] == 0:
                row["disease_has_no_lesion"] = True
                issues.append("disease_has_no_lesion")

            row["issues"] = ";".join(issues)
            row["status"] = "FAIL" if issues else "OK"
            rows.append(row)

    df = pd.DataFrame(rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT, index=False)

    print("=" * 70)
    print("LABELME ANNOTATION QA")
    print("=" * 70)
    print(f"Images checked: {len(df)}")
    print(f"PASS: {(df.status == 'OK').sum()}")
    print(f"NEEDS REVIEW: {(df.status != 'OK').sum()}")
    print()
    print("By class:")
    print(df.groupby(["class", "status"]).size().unstack(fill_value=0).to_string())
    print()
    if len(df):
        issue_counts = (
            df.assign(issue=df["issues"].str.split(";"))
              .explode("issue")
        )
        issue_counts = issue_counts[
            issue_counts["issue"].notna() &
            (issue_counts["issue"] != "")
        ]["issue"].value_counts()
        print("Issues:")
        print(issue_counts.to_string() if len(issue_counts) else "None")
    print()
    print(f"Report: {REPORT}")

if __name__ == "__main__":
    main()
