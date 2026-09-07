from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
import math

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_ROOT / "outputs" / "metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "severity_pilot"

N_PER_CLASS = 20
SEED = 42

# 5 columns x 4 rows = 20 images
COLS = 5
ROWS = 4

THUMB_SIZE = 256
LABEL_HEIGHT = 45
MARGIN = 10

# ============================================================
# Setup
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)

df = pd.read_csv(METADATA_PATH)

# Remove duplicate records
df = df[df["duplicate"] == False].copy()

classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]

class_names = {
    "Apple___Apple_scab": "Apple Scab",
    "Apple___Black_rot": "Black Rot",
    "Apple___Cedar_apple_rust": "Cedar Apple Rust",
    "Apple___healthy": "Healthy",
}

# ============================================================
# Locate images
# ============================================================

def resolve_image_path(relative_path):
    """
    metadata.csv contains paths such as:
    dataset\\Apple___Apple_scab\\image (1).JPG
    """

    path = PROJECT_ROOT / Path(str(relative_path).replace("\\", "/"))

    return path


# ============================================================
# Font
# ============================================================

try:
    font = ImageFont.truetype("arial.ttf", 18)
    title_font = ImageFont.truetype("arial.ttf", 28)
except:
    font = ImageFont.load_default()
    title_font = font


# ============================================================
# Create one contact sheet per class
# ============================================================

for class_name in classes:

    class_df = df[df["class"] == class_name].copy()

    if len(class_df) < N_PER_CLASS:
        print(
            f"WARNING: {class_name} has only "
            f"{len(class_df)} images."
        )
        sample_df = class_df
    else:
        sample_df = class_df.sample(
            n=N_PER_CLASS,
            random_state=SEED
        )

    images = []

    for _, row in sample_df.iterrows():

        img_path = resolve_image_path(row["image_path"])

        if not img_path.exists():
            print(f"Missing: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((THUMB_SIZE, THUMB_SIZE))

            # White canvas so all thumbnails have same dimensions
            canvas = Image.new(
                "RGB",
                (THUMB_SIZE, THUMB_SIZE),
                "white"
            )

            x = (THUMB_SIZE - img.width) // 2
            y = (THUMB_SIZE - img.height) // 2

            canvas.paste(img, (x, y))

            images.append(
                (
                    canvas,
                    Path(row["image_path"]).name
                )
            )

        except Exception as e:
            print(f"Could not read {img_path}: {e}")

    # --------------------------------------------------------
    # Sheet dimensions
    # --------------------------------------------------------

    cell_width = THUMB_SIZE + 2 * MARGIN
    cell_height = THUMB_SIZE + LABEL_HEIGHT + 2 * MARGIN

    title_height = 60

    sheet_width = COLS * cell_width
    sheet_height = title_height + ROWS * cell_height

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    title = class_names[class_name]

    draw.text(
        (MARGIN, 15),
        f"{title} — {len(images)} sample images",
        fill="black",
        font=title_font
    )

    # --------------------------------------------------------
    # Paste images
    # --------------------------------------------------------

    for i, (img, filename) in enumerate(images):

        row_idx = i // COLS
        col_idx = i % COLS

        x = col_idx * cell_width + MARGIN
        y = title_height + row_idx * cell_height + MARGIN

        sheet.paste(img, (x, y))

        # Shorten very long filenames
        display_name = filename

        if len(display_name) > 30:
            display_name = display_name[:27] + "..."

        draw.text(
            (x, y + THUMB_SIZE + 5),
            display_name,
            fill="black",
            font=font
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    safe_name = class_name.replace("Apple___", "").lower()

    output_path = OUTPUT_DIR / f"{safe_name}_pilot.jpg"

    sheet.save(
        output_path,
        quality=95
    )

    print(f"Created: {output_path}")


print("\nDone.")
print(f"Output folder: {OUTPUT_DIR}")