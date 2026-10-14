from pathlib import Path
import math
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"E:\New Volume D\23BLC1297\apple_disease_project")

GROUND_TRUTH_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "severity"
    / "severity_ground_truth.csv"
)

IMAGE_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "severity"
    / "annotation_images"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "severity"
    / "grade_visualizations"
)

# Number of representative images per disease/grade
IMAGES_PER_GRADE = 3

# Size of each image in the contact sheet
IMAGE_SIZE = 300

# Padding between cells
PADDING = 25

# ------------------------------------------------------------
# Four severity grades
# ------------------------------------------------------------

GRADE_BINS = {
    0: (0.0, 5.0),
    1: (5.0, 15.0),
    2: (15.0, 30.0),
    3: (30.0, 100.0),
}

GRADE_NAMES = {
    0: "Grade 0",
    1: "Grade 1",
    2: "Grade 2",
    3: "Grade 3",
}

GRADE_DESCRIPTIONS = {
    0: "0–5%",
    1: ">5–15%",
    2: ">15–30%",
    3: ">30–100%",
}

DISEASES = [
    "apple_scab",
    "black_rot",
    "cedar_rust",
    "healthy",
]

DISPLAY_NAMES = {
    "apple_scab": "Apple Scab",
    "black_rot": "Black Rot",
    "cedar_rust": "Cedar Apple Rust",
    "healthy": "Healthy",
}


# ============================================================
# HELPERS
# ============================================================

def assign_grade(si):
    """Assign one of the four severity grades."""

    if si <= 5:
        return 0
    elif si <= 15:
        return 1
    elif si <= 30:
        return 2
    else:
        return 3


def get_image_path(mask_path):
    """
    Convert a mask path such as:

        masks/apple_scab/image (5).png

    into the corresponding annotation image:

        annotation_images/apple_scab/image (5).JPG

    Handles different image extensions.
    """

    mask_path = Path(mask_path)

    # Extract disease class from path
    disease = mask_path.parent.name

    stem = mask_path.stem

    disease_dir = IMAGE_ROOT / disease

    if not disease_dir.exists():
        return None

    # Look for matching filename regardless of extension
    candidates = list(disease_dir.glob(stem + ".*"))

    # Prefer common image formats
    preferred_extensions = [
        ".JPG",
        ".jpg",
        ".JPEG",
        ".jpeg",
        ".PNG",
        ".png",
    ]

    for ext in preferred_extensions:
        for candidate in candidates:
            if candidate.suffix == ext:
                return candidate

    if candidates:
        return candidates[0]

    return None


def load_font(size):
    """Try to load a useful font; fall back to PIL default."""

    font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]

    for font_path in font_candidates:
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)

    return ImageFont.load_default()


def resize_and_crop(image, size):
    """
    Resize while preserving aspect ratio and center-crop.
    """

    image = image.convert("RGB")

    width, height = image.size

    scale = max(size / width, size / height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    left = (new_width - size) // 2
    top = (new_height - size) // 2

    image = image.crop(
        (left, top, left + size, top + size)
    )

    return image


def select_representative_images(group, n):
    """
    Select images closest to the median SI for that
    disease/grade.

    This gives visually representative examples instead
    of simply selecting the first files.
    """

    group = group.copy()

    median_si = group["severity_index_percent"].median()

    group["distance_from_median"] = (
        group["severity_index_percent"] - median_si
    ).abs()

    selected = group.sort_values(
        "distance_from_median"
    ).head(n)

    return selected


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_dataframe():

    df = pd.read_csv(GROUND_TRUTH_CSV)

    required_columns = [
        "mask_path",
        "severity_index_percent",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Infer disease from mask path
    df["disease"] = (
        df["mask_path"]
        .astype(str)
        .apply(lambda x: Path(x).parent.name)
    )

    # Assign four-grade severity
    df["grade"] = (
        df["severity_index_percent"]
        .apply(assign_grade)
    )

    # Find corresponding original image
    df["image_path"] = df["mask_path"].apply(
        get_image_path
    )

    missing_images = df["image_path"].isna().sum()

    if missing_images > 0:
        print(
            f"WARNING: {missing_images} images could not be located."
        )

    df = df.dropna(subset=["image_path"])

    return df


# ============================================================
# CREATE ONE DISEASE × GRADE CONTACT SHEET
# ============================================================

def create_disease_sheet(df, disease):

    disease_df = df[df["disease"] == disease]

    selected = []

    for grade in range(4):

        grade_df = disease_df[
            disease_df["grade"] == grade
        ]

        if len(grade_df) == 0:
            continue

        examples = select_representative_images(
            grade_df,
            IMAGES_PER_GRADE
        )

        for _, row in examples.iterrows():
            selected.append(row)

    if not selected:
        return

    rows = 4
    cols = IMAGES_PER_GRADE

    title_height = 80
    label_height = 55

    sheet_width = (
        cols * IMAGE_SIZE
        + (cols + 1) * PADDING
    )

    sheet_height = (
        title_height
        + rows * (
            IMAGE_SIZE
            + label_height
            + PADDING
        )
        + PADDING
    )

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    title_font = load_font(28)
    label_font = load_font(20)

    title = (
        f"{DISPLAY_NAMES[disease]} — "
        f"Severity Grade Examples"
    )

    draw.text(
        (PADDING, 20),
        title,
        fill="black",
        font=title_font
    )

    # Group selected images by grade
    for grade in range(4):

        grade_examples = [
            row for row in selected
            if row["grade"] == grade
        ]

        for col, row in enumerate(grade_examples):

            image = Image.open(
                row["image_path"]
            )

            image = resize_and_crop(
                image,
                IMAGE_SIZE
            )

            x = (
                PADDING
                + col * (
                    IMAGE_SIZE + PADDING
                )
            )

            y = (
                title_height
                + grade * (
                    IMAGE_SIZE
                    + label_height
                    + PADDING
                )
            )

            sheet.paste(image, (x, y))

            si = row["severity_index_percent"]

            label = (
                f"{GRADE_NAMES[grade]} "
                f"({GRADE_DESCRIPTIONS[grade]})\n"
                f"SI = {si:.2f}%"
            )

            draw.multiline_text(
                (x, y + IMAGE_SIZE + 8),
                label,
                fill="black",
                font=label_font,
                spacing=4,
            )

    output_path = (
        OUTPUT_ROOT
        / f"{disease}_severity_grades.png"
    )

    sheet.save(
        output_path,
        quality=95
    )

    print(f"Saved: {output_path}")


# ============================================================
# CREATE MASTER SHEET
# ============================================================

def create_master_sheet(df):

    # One representative image per disease × grade
    selected = []

    for disease in DISEASES:

        disease_df = df[
            df["disease"] == disease
        ]

        for grade in range(4):

            grade_df = disease_df[
                disease_df["grade"] == grade
            ]

            if len(grade_df) == 0:
                continue

            example = select_representative_images(
                grade_df,
                1
            )

            if len(example) > 0:
                selected.append(
                    example.iloc[0]
                )

    rows = 4
    cols = 4

    title_height = 100
    label_height = 60

    sheet_width = (
        cols * IMAGE_SIZE
        + (cols + 1) * PADDING
    )

    sheet_height = (
        title_height
        + rows * (
            IMAGE_SIZE
            + label_height
            + PADDING
        )
        + PADDING
    )

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    title_font = load_font(30)
    label_font = load_font(18)

    draw.text(
        (PADDING, 20),
        "Apple Disease Severity Grade Examples",
        fill="black",
        font=title_font
    )

    draw.text(
        (PADDING, 55),
        "Representative images selected near the median SI "
        "within each disease/grade",
        fill="black",
        font=label_font
    )

    # Fixed layout:
    # rows = diseases
    # columns = grades

    for disease_idx, disease in enumerate(DISEASES):

        for grade in range(4):

            candidates = df[
                (df["disease"] == disease)
                & (df["grade"] == grade)
            ]

            if len(candidates) == 0:
                continue

            example = select_representative_images(
                candidates,
                1
            ).iloc[0]

            image = Image.open(
                example["image_path"]
            )

            image = resize_and_crop(
                image,
                IMAGE_SIZE
            )

            x = (
                PADDING
                + grade * (
                    IMAGE_SIZE + PADDING
                )
            )

            y = (
                title_height
                + disease_idx * (
                    IMAGE_SIZE
                    + label_height
                    + PADDING
                )
            )

            sheet.paste(image, (x, y))

            si = example[
                "severity_index_percent"
            ]

            label = (
                f"{DISPLAY_NAMES[disease]}\n"
                f"Grade {grade} "
                f"({GRADE_DESCRIPTIONS[grade]})\n"
                f"SI = {si:.2f}%"
            )

            draw.multiline_text(
                (x, y + IMAGE_SIZE + 8),
                label,
                fill="black",
                font=label_font,
                spacing=3,
            )

    output_path = (
        OUTPUT_ROOT
        / "all_diseases_all_grades.png"
    )

    sheet.save(
        output_path,
        quality=95
    )

    print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading ground-truth severity data...")

    df = prepare_dataframe()

    print(f"Images available: {len(df)}")

    print("\nGrade distribution:")

    print(
        df["grade"]
        .value_counts()
        .sort_index()
    )

    print("\nDisease × Grade distribution:")

    distribution = pd.crosstab(
        df["disease"],
        df["grade"]
    )

    print(distribution)

    print("\nCreating disease-specific sheets...")

    for disease in DISEASES:
        create_disease_sheet(
            df,
            disease
        )

    print("\nCreating master sheet...")

    create_master_sheet(df)

    print("\nDone.")


if __name__ == "__main__":
    main()