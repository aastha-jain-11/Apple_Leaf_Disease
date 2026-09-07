from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASK_ROOT = PROJECT_ROOT / "outputs" / "severity" / "masks"
OUT_CSV = PROJECT_ROOT / "outputs" / "severity" / "severity_ground_truth.csv"


def main():
    mask_files = list(MASK_ROOT.glob("**/*.png"))
    if not mask_files:
        print(f"No masks found in {MASK_ROOT}")
        return

    rows = []

    for path in mask_files:
        mask = np.asarray(Image.open(path))

        leaf_pixels = int(np.sum(mask == 1))
        lesion_pixels = int(np.sum(mask == 2))

        if leaf_pixels == 0:
            si = np.nan
        else:
            si = (lesion_pixels / leaf_pixels) * 100.0

        # The grade is intentionally NOT assigned here yet.
        # We will finalize six-grade thresholds after inspecting the
        # real SI distribution and validating the scale.
        rows.append({
            "mask_path": str(path.relative_to(PROJECT_ROOT)),
            "leaf_pixels": leaf_pixels,
            "lesion_pixels": lesion_pixels,
            "severity_index_percent": si,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("=" * 70)
    print("SEVERITY INDEX CALCULATION")
    print("=" * 70)
    print(f"Masks processed : {len(df)}")
    print(f"Output CSV      : {OUT_CSV}")
    print()
    print(df["severity_index_percent"].describe().to_string())


if __name__ == "__main__":
    main()
