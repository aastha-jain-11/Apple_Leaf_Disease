from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from severity_config import IMAGE_SIZE, NUM_CLASSES

MODEL_DIR = PROJECT_ROOT / "models" / "segformer" / "severity" / "best"
SPLIT_CSV = PROJECT_ROOT / "outputs" / "severity" / "severity_segmentation_split.csv"
IMAGE_ROOT = PROJECT_ROOT / "outputs" / "severity" / "annotation_images"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "severity" / "predictions"
FEATURE_CSV = PROJECT_ROOT / "outputs" / "severity" / "severity_features_predicted.csv"
MIN_LESION_AREA_PIXELS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = (arr - np.array([0.485,0.456,0.406],np.float32)) / np.array([0.229,0.224,0.225],np.float32)
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)

def resolve_image_path(row):
    filename = Path(str(row["filename"]))
    class_name = str(row["class"])

    class_dir = IMAGE_ROOT / class_name

    # Try the exact filename first.
    exact_path = class_dir / filename.name
    if exact_path.is_file():
        return exact_path

    # The split stores .png names, while annotation images may be .JPG/.jpg.
    # Match using the filename stem.
    stem = filename.stem

    for path in class_dir.iterdir():
        if path.is_file() and path.stem == stem:
            return path

    raise FileNotFoundError(
        f"Could not locate image with stem '{stem}' in {class_dir}"
    )

def extract_features(mask):
    leaf = mask == 1
    lesion = mask == 2
    leaf_pixels = int(leaf.sum())
    lesion_pixels = int(lesion.sum())
    if leaf_pixels == 0:
        return dict(leaf_pixels=0, lesion_pixels=lesion_pixels,
                    lesion_area_percent=np.nan, lesion_count=0,
                    mean_lesion_area=0.0, max_lesion_area=0.0,
                    lesion_area_std=0.0)
    binary = lesion.astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    areas = areas[areas >= MIN_LESION_AREA_PIXELS].astype(float)
    return dict(
        leaf_pixels=leaf_pixels,
        lesion_pixels=lesion_pixels,
        lesion_area_percent=100.0 * lesion_pixels / (leaf_pixels + lesion_pixels),
        lesion_count=int(len(areas)),
        mean_lesion_area=float(areas.mean()) if len(areas) else 0.0,
        max_lesion_area=float(areas.max()) if len(areas) else 0.0,
        lesion_area_std=float(areas.std()) if len(areas) else 0.0,
    )

def main():
    print("="*70)
    print("SEGFORMER SEVERITY FEATURE EXTRACTION")
    print("="*70)
    print("Device:", DEVICE)
    print("Model:", MODEL_DIR)
    print("Minimum lesion component:", MIN_LESION_AREA_PIXELS, "pixels")

    df = pd.read_csv(SPLIT_CSV)
    model = SegformerForSemanticSegmentation.from_pretrained(
        str(MODEL_DIR), num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
    ).to(DEVICE)
    model.eval()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    records = []
    with torch.no_grad():
        for i, row in df.iterrows():
            image_path = resolve_image_path(row)
            x = load_image(image_path).to(DEVICE)
            logits = F.interpolate(
                model(pixel_values=x).logits,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear", align_corners=False
            )
            mask = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            features = extract_features(mask)

            class_dir = OUTPUT_ROOT / str(row["class"])
            class_dir.mkdir(parents=True, exist_ok=True)
            mask_path = class_dir / f"{image_path.stem}_pred_mask.png"
            cv2.imwrite(str(mask_path), mask)

            records.append({
                "filename": str(row["filename"]),
                "class": str(row["class"]),
                "split": str(row["split"]) if "split" in row.index else "",
                "predicted_mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
                **features
            })
            if (i+1) % 50 == 0 or i+1 == len(df):
                print(f"Processed {i+1}/{len(df)}")

    out = pd.DataFrame(records)
    out.to_csv(FEATURE_CSV, index=False)
    print("\nCompleted.")
    print("Feature CSV:", FEATURE_CSV)
    print("Predicted masks:", OUTPUT_ROOT)
    print("\nLesion-count summary:")
    print(out["lesion_count"].describe().to_string())
    print("\nSeverity-index summary:")
    print(out["lesion_area_percent"].describe().to_string())

if __name__ == "__main__":
    main()
