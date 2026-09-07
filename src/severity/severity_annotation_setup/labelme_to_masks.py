from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_ROOT = PROJECT_ROOT / "outputs" / "severity" / "annotation_images"
JSON_ROOT = PROJECT_ROOT / "outputs" / "severity" / "labels_json"
MASK_ROOT = PROJECT_ROOT / "outputs" / "severity" / "masks"

# Pixel classes:
# 0 = background
# 1 = leaf
# 2 = lesion
#
# Background is implicit: any pixel not covered by a leaf polygon
# remains 0. If a lesion overlaps a leaf polygon, lesion wins.


def polygon_to_xy(points):
    return [(int(round(x)), int(round(y))) for x, y in points]


def convert_one(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    image_name = data.get("imagePath", json_path.stem)
    image_path = IMAGE_ROOT / json_path.parent.name / Path(image_name).name

    if not image_path.exists():
        # Search class folders if imagePath has no useful class information.
        matches = list(IMAGE_ROOT.glob(f"*/{Path(image_name).name}"))
        if not matches:
            raise FileNotFoundError(f"Image not found for {json_path}: {image_name}")
        image_path = matches[0]

    with Image.open(image_path) as im:
        width, height = im.size

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # First draw leaf, then lesion so lesion always overrides leaf.
    shapes = data.get("shapes", [])

    for shape in shapes:
        label = str(shape.get("label", "")).strip().lower()
        if label != "leaf":
            continue
        if shape.get("shape_type") != "polygon":
            continue
        draw.polygon(polygon_to_xy(shape["points"]), fill=1)

    for shape in shapes:
        label = str(shape.get("label", "")).strip().lower()
        if label != "lesion":
            continue
        if shape.get("shape_type") != "polygon":
            continue
        draw.polygon(polygon_to_xy(shape["points"]), fill=2)

    relative_class = json_path.parent.name
    out_dir = MASK_ROOT / relative_class
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{json_path.stem}.png"
    mask.save(out_path)
    return out_path


def main():
    if not JSON_ROOT.exists():
        raise FileNotFoundError(f"JSON folder not found: {JSON_ROOT}")

    json_files = list(JSON_ROOT.glob("**/*.json"))
    if not json_files:
        print(f"No LabelMe JSON files found in {JSON_ROOT}")
        return

    converted = 0
    errors = []

    for json_path in json_files:
        try:
            convert_one(json_path)
            converted += 1
        except Exception as exc:
            errors.append((str(json_path), str(exc)))

    print("=" * 70)
    print("LABELME JSON -> 3-CLASS MASK CONVERSION")
    print("=" * 70)
    print(f"JSON files : {len(json_files)}")
    print(f"Converted  : {converted}")
    print(f"Errors     : {len(errors)}")

    if errors:
        error_path = PROJECT_ROOT / "outputs" / "severity" / "mask_conversion_errors.csv"
        import pandas as pd
        pd.DataFrame(errors, columns=["json_path", "error"]).to_csv(
            error_path, index=False
        )
        print(f"Error list : {error_path}")


if __name__ == "__main__":
    main()
