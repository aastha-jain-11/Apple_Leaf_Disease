from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = PROJECT_ROOT / "outputs" / "severity" / "annotation_images"
MASK_ROOT = PROJECT_ROOT / "outputs" / "severity" / "masks"
OUT_CSV = PROJECT_ROOT / "outputs" / "severity" / "severity_ground_truth.csv"

def xy(points):
    return [(int(round(x)), int(round(y))) for x, y in points]

def convert(json_path):
    class_name = json_path.parent.name
    image_path = None
    for ext in [".JPG",".jpg",".JPEG",".jpeg",".PNG",".png"]:
        p = json_path.with_suffix(ext)
        if p.exists():
            image_path = p
            break
    if image_path is None:
        raise FileNotFoundError(f"No matching image for {json_path}")

    with Image.open(image_path) as im:
        size = im.size

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Leaf first.
    for s in data.get("shapes", []):
        if str(s.get("label","")).strip().lower() == "leaf" and s.get("shape_type") == "polygon":
            draw.polygon(xy(s["points"]), fill=1)

    # Lesion second so lesion wins wherever polygons overlap.
    for s in data.get("shapes", []):
        if str(s.get("label","")).strip().lower() == "lesion" and s.get("shape_type") == "polygon":
            draw.polygon(xy(s["points"]), fill=2)

    out = MASK_ROOT / class_name / f"{json_path.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    mask.save(out)

def main():
    json_files = list(IMAGE_ROOT.glob("**/*.json"))
    print(f"JSON files found: {len(json_files)}")

    errors = []
    for p in json_files:
        try:
            convert(p)
        except Exception as e:
            errors.append((str(p), str(e)))

    print(f"Converted: {len(json_files) - len(errors)}")
    print(f"Errors: {len(errors)}")

    if errors:
        pd.DataFrame(errors, columns=["json_path","error"]).to_csv(
            PROJECT_ROOT / "outputs" / "severity" / "mask_conversion_errors.csv",
            index=False
        )

if __name__ == "__main__":
    main()
