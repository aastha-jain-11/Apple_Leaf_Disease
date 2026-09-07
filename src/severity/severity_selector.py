"""
Apple Disease Severity Candidate Selector

Place this file at:
D:/23BLC1297/apple_disease_project/src/severity/severity_selector.py

Input:
D:/23BLC1297/apple_disease_project/metadata.csv

Output:
D:/23BLC1297/apple_disease_project/outputs/severity_selection.csv
"""

import csv
import random
import re
import sys
from pathlib import Path

import pandas as pd
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_FILE = PROJECT_ROOT / "metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "severity_selection.csv"


# ============================================================
# REVIEW POOL
# ============================================================
# These are candidate images to REVIEW.
# They are NOT the final severity dataset.

SAMPLE_PER_CLASS = {
    "apple_scab": 200,
    "black_rot": 200,
    "cedar_rust": 200,
    "healthy": 100,
}

RANDOM_SEED = 42

MAX_IMAGE_W = 760
MAX_IMAGE_H = 650


# ============================================================
# CLASS NORMALIZATION
# ============================================================

def normalize_class_name(value):
    """
    Convert all likely metadata naming formats into one of:

        apple_scab
        black_rot
        cedar_rust
        healthy

    Handles formats such as:
        Apple Scab
        Apple___Apple_scab
        Apple_Apple_scab
        Black Rot
        Apple___Black_rot
        Cedar Apple Rust
        Apple___Cedar_apple_rust
        Healthy
        Apple___healthy
    """
    text = str(value).strip().lower()

    # Replace every non-alphanumeric character with a space.
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if "healthy" in text:
        return "healthy"

    if "black" in text and "rot" in text:
        return "black_rot"

    if "scab" in text:
        return "apple_scab"

    if "cedar" in text and "rust" in text:
        return "cedar_rust"

    if "rust" in text and "apple" in text:
        return "cedar_rust"

    return "unknown"


# ============================================================
# IMAGE PATH RESOLUTION
# ============================================================

def resolve_image_path(image_path_value):
    """
    Resolve the image path stored in metadata.csv.

    Expected metadata paths are relative to the project root,
    for example:
        dataset\\Apple___Apple_scab\\image (1).JPG
    """
    raw = str(image_path_value).strip()

    if not raw:
        return None

    # Normalize both slash styles.
    normalized = raw.replace("/", "\\")

    # If metadata somehow contains an absolute Windows path.
    absolute_candidate = Path(normalized)
    if absolute_candidate.is_absolute() and absolute_candidate.exists():
        return absolute_candidate

    # Normal project-relative path.
    candidate = PROJECT_ROOT / Path(normalized)
    if candidate.exists():
        return candidate

    # Also try stripping ./ or .\
    cleaned = normalized.lstrip(".\\/ ")
    candidate = PROJECT_ROOT / Path(cleaned)

    if candidate.exists():
        return candidate

    return None


# ============================================================
# METADATA
# ============================================================

def load_metadata():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"metadata.csv was not found:\n{METADATA_FILE}"
        )

    df = pd.read_csv(METADATA_FILE)

    required_columns = {"image_path", "class"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"metadata.csv is missing columns: {sorted(missing)}"
        )

    # Remove duplicate records if duplicate information exists.
    if "duplicate" in df.columns:
        duplicate_text = (
            df["duplicate"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        duplicate_mask = duplicate_text.isin(
            {"true", "1", "yes", "y"}
        )

        df = df.loc[~duplicate_mask].copy()

    df["internal_class"] = df["class"].map(normalize_class_name)

    return df


# ============================================================
# CANDIDATE POOL
# ============================================================

def build_candidate_pool(df):
    rng = random.Random(RANDOM_SEED)

    # Diagnostic output: this prevents silent class-name failures.
    print("Raw metadata class values:")
    for value in sorted(df["class"].astype(str).unique()):
        print(f"  {value}")

    print("\nNormalized class counts:")
    counts = df["internal_class"].value_counts(dropna=False)
    for name, count in counts.items():
        print(f"  {name}: {count}")

    print()

    candidates = []

    for class_name, requested_count in SAMPLE_PER_CLASS.items():
        subset = df[df["internal_class"] == class_name].copy()

        if subset.empty:
            print(
                f"ERROR: No images found for normalized class "
                f"'{class_name}'."
            )
            continue

        records = subset.to_dict("records")
        rng.shuffle(records)

        selected = records[:min(requested_count, len(records))]

        found = 0

        for record in selected:
            image_path = resolve_image_path(record["image_path"])

            if image_path is None:
                print(
                    "WARNING: Image file not found for metadata path: "
                    f"{record['image_path']}"
                )
                continue

            candidates.append({
                "image_path": str(record["image_path"]),
                "class": str(record["class"]),
                "internal_class": class_name,
                "selection": "",
                "reviewed": 0,
            })

            found += 1

        print(
            f"{class_name}: selected {found} valid images "
            f"from requested pool of {requested_count}"
        )

    rng.shuffle(candidates)

    return candidates


# ============================================================
# EXISTING RESULTS
# ============================================================

def load_previous_results():
    """
    Read previously saved decisions so the selector can resume.
    """
    if not OUTPUT_FILE.exists():
        return {}

    try:
        old = pd.read_csv(OUTPUT_FILE)
    except Exception:
        return {}

    previous = {}

    if "image_path" not in old.columns:
        return previous

    for _, row in old.iterrows():
        path = str(row.get("image_path", "")).strip()
        selection = str(row.get("selection", "")).strip()

        if path and selection:
            previous[path] = selection

    return previous


# ============================================================
# GUI
# ============================================================

class SeveritySelector:

    def __init__(self, root, candidates):
        self.root = root
        self.candidates = candidates

        self.previous = load_previous_results()

        # Resume by skipping already decided images.
        self.remaining = [
            item
            for item in candidates
            if item["image_path"] not in self.previous
        ]

        self.index = 0
        self.history = []
        self.current_photo = None

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.root.title(
            "Apple Disease Severity Candidate Selector"
        )
        self.root.geometry("1050x850")
        self.root.minsize(900, 700)

        self.build_ui()
        self.bind_keys()

        if self.remaining:
            self.show_current()
        else:
            self.show_finished()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def build_ui(self):
        header = tk.Frame(self.root)
        header.pack(fill="x", padx=12, pady=(10, 4))

        title = tk.Label(
            header,
            text="Apple Disease Severity Candidate Selector",
            font=("Arial", 18, "bold"),
        )
        title.pack()

        self.info_label = tk.Label(
            header,
            text="",
            font=("Arial", 11),
        )
        self.info_label.pack(pady=(4, 0))

        self.image_frame = tk.Frame(
            self.root,
            relief="sunken",
            borderwidth=1,
        )
        self.image_frame.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=8,
        )

        self.image_label = tk.Label(
            self.image_frame,
            text="Loading...",
            font=("Arial", 14),
        )
        self.image_label.pack(
            fill="both",
            expand=True,
        )

        self.path_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 9),
            wraplength=950,
        )
        self.path_label.pack(
            padx=12,
            pady=(0, 4),
        )

        button_frame = tk.Frame(self.root)
        button_frame.pack(
            fill="x",
            padx=12,
            pady=8,
        )

        buttons = [
            ("1  LOW", "low"),
            ("2  MODERATE", "moderate"),
            ("3  HIGH", "high"),
            ("X  EXCLUDE", "exclude"),
            ("SPACE  SKIP", "skip"),
            ("U  UNDO", "undo"),
        ]

        for text, command in buttons:
            button = tk.Button(
                button_frame,
                text=text,
                command=lambda c=command: self.handle_decision(c),
                font=("Arial", 11, "bold"),
                padx=8,
                pady=8,
            )
            button.pack(
                side="left",
                expand=True,
                fill="x",
                padx=3,
            )

        footer = tk.Label(
            self.root,
            text=(
                "1=Low   2=Moderate   3=High   "
                "X=Exclude   Space/N=Skip   U=Undo   "
                "Esc=Save & Exit"
            ),
            font=("Arial", 10),
        )
        footer.pack(pady=(0, 10))

    def bind_keys(self):
        self.root.bind(
            "<KeyPress-1>",
            lambda event: self.handle_decision("low"),
        )
        self.root.bind(
            "<KeyPress-2>",
            lambda event: self.handle_decision("moderate"),
        )
        self.root.bind(
            "<KeyPress-3>",
            lambda event: self.handle_decision("high"),
        )
        self.root.bind(
            "<KeyPress-x>",
            lambda event: self.handle_decision("exclude"),
        )
        self.root.bind(
            "<KeyPress-X>",
            lambda event: self.handle_decision("exclude"),
        )
        self.root.bind(
            "<space>",
            lambda event: self.handle_decision("skip"),
        )
        self.root.bind(
            "<KeyPress-n>",
            lambda event: self.handle_decision("skip"),
        )
        self.root.bind(
            "<KeyPress-N>",
            lambda event: self.handle_decision("skip"),
        )
        self.root.bind(
            "<KeyPress-u>",
            lambda event: self.undo(),
        )
        self.root.bind(
            "<KeyPress-U>",
            lambda event: self.undo(),
        )
        self.root.bind(
            "<Escape>",
            lambda event: self.close(),
        )

    # --------------------------------------------------------
    # IMAGE DISPLAY
    # --------------------------------------------------------

    def show_current(self):
        if self.index >= len(self.remaining):
            self.show_finished()
            return

        item = self.remaining[self.index]

        image_path = resolve_image_path(item["image_path"])

        if image_path is None:
            self.index += 1
            self.show_current()
            return

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            self.path_label.config(
                text=f"Could not open image:\n{image_path}\n{exc}"
            )
            self.index += 1
            self.show_current()
            return

        image.thumbnail(
            (MAX_IMAGE_W, MAX_IMAGE_H),
            Image.Resampling.LANCZOS,
        )

        self.current_photo = ImageTk.PhotoImage(image)

        self.image_label.config(
            image=self.current_photo,
            text="",
        )

        display_class = (
            item["internal_class"]
            .replace("_", " ")
            .title()
        )

        self.info_label.config(
            text=(
                f"Candidate {self.index + 1} / {len(self.remaining)}"
                f"     |     Class: {display_class}"
            )
        )

        self.path_label.config(
            text=str(image_path)
        )

    # --------------------------------------------------------
    # DECISIONS
    # --------------------------------------------------------

    def handle_decision(self, decision):
        if self.index >= len(self.remaining):
            return

        item = self.remaining[self.index]

        # Healthy images are not disease-severity levels.
        # For healthy images:
        #   1 = keep as healthy candidate
        #   X = exclude
        if item["internal_class"] == "healthy":
            if decision in {"moderate", "high"}:
                messagebox.showinfo(
                    "Healthy image",
                    "This is a healthy image.\n\n"
                    "Use 1 to keep it as a healthy candidate, "
                    "or X to exclude it."
                )
                return

        item["selection"] = decision
        item["reviewed"] = 1

        self.history.append(
            (self.index, item.copy())
        )

        self.save_progress_row(item)

        self.index += 1
        self.show_current()

    # --------------------------------------------------------
    # CSV SAVE
    # --------------------------------------------------------

    def save_progress_row(self, item):
        existing = []

        if OUTPUT_FILE.exists():
            try:
                with OUTPUT_FILE.open(
                    "r",
                    newline="",
                    encoding="utf-8-sig",
                ) as file:
                    existing = list(csv.DictReader(file))
            except Exception:
                existing = []

        updated = False

        for row in existing:
            if row.get("image_path") == item["image_path"]:
                row.update({
                    "class": item["class"],
                    "internal_class": item["internal_class"],
                    "selection": item["selection"],
                    "reviewed": "1",
                })
                updated = True
                break

        if not updated:
            existing.append({
                "image_path": item["image_path"],
                "class": item["class"],
                "internal_class": item["internal_class"],
                "selection": item["selection"],
                "reviewed": "1",
            })

        with OUTPUT_FILE.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "image_path",
                    "class",
                    "internal_class",
                    "selection",
                    "reviewed",
                ],
            )
            writer.writeheader()
            writer.writerows(existing)

    # --------------------------------------------------------
    # UNDO
    # --------------------------------------------------------

    def undo(self):
        if not self.history:
            return

        previous_index, item = self.history.pop()

        self.remove_saved_row(
            item["image_path"]
        )

        self.index = previous_index
        self.show_current()

    def remove_saved_row(self, image_path):
        if not OUTPUT_FILE.exists():
            return

        try:
            with OUTPUT_FILE.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                rows = list(csv.DictReader(file))

            rows = [
                row
                for row in rows
                if row.get("image_path") != image_path
            ]

            with OUTPUT_FILE.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "image_path",
                        "class",
                        "internal_class",
                        "selection",
                        "reviewed",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)

        except Exception:
            pass

    # --------------------------------------------------------
    # FINISH / CLOSE
    # --------------------------------------------------------

    def show_finished(self):
        self.image_label.config(
            image="",
            text=(
                "Review complete.\n\n"
                "Results saved to:\n"
                f"{OUTPUT_FILE}"
            ),
            font=("Arial", 18, "bold"),
        )

        self.info_label.config(
            text="All candidates in this review pool have been processed."
        )

        self.path_label.config(text="")

    def close(self):
        self.root.destroy()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("APPLE DISEASE SEVERITY CANDIDATE SELECTOR")
    print("=" * 70)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Metadata     : {METADATA_FILE}")
    print(f"Output       : {OUTPUT_FILE}")
    print()

    try:
        df = load_metadata()

        candidates = build_candidate_pool(df)

        if not candidates:
            raise RuntimeError(
                "No valid candidate images were found."
            )

        print()
        print(f"Total valid candidates: {len(candidates)}")
        print()

    except Exception as exc:
        print(f"ERROR: {exc}")
        input("Press Enter to exit...")
        sys.exit(1)

    root = tk.Tk()
    SeveritySelector(root, candidates)
    root.mainloop()


if __name__ == "__main__":
    main()
