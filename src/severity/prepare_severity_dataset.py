from pathlib import Path
import random
import pandas as pd
from severity_config import MASKS, SEVERITY_ROOT, SPLIT_CSV, SEED

CLASSES = ["apple_scab", "black_rot", "cedar_rust", "healthy"]

def main():
    masks = list(MASKS.glob("**/*.png"))
    records = []
    for p in masks:
        if p.parent.name in CLASSES:
            records.append({
                "mask_path": str(p.relative_to(SEVERITY_ROOT)),
                "class": p.parent.name,
                "filename": p.name
            })
    if not records:
        raise RuntimeError(f"No masks found under {MASKS}.")
    df = pd.DataFrame(records)
    rng = random.Random(SEED)
    df["split"] = ""
    for cls, g in df.groupby("class"):
        idx = list(g.index)
        rng.shuffle(idx)
        n = len(idx)
        a = int(round(n * 0.70))
        b = int(round(n * 0.15))
        df.loc[idx[:a], "split"] = "train"
        df.loc[idx[a:a+b], "split"] = "validation"
        df.loc[idx[a+b:], "split"] = "test"
    df = df.sort_values(["class","split","filename"]).reset_index(drop=True)
    df.to_csv(SPLIT_CSV, index=False)
    print(df.groupby(["class","split"]).size().unstack(fill_value=0))
    print(f"Saved: {SPLIT_CSV}")

if __name__ == "__main__":
    main()
