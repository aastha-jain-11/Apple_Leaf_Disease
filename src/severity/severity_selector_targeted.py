"""
Targeted second-pass selector for the Apple Disease Severity project.

PURPOSE
-------
The first review round produced an uneven visible-severity distribution:

    Apple Scab     : 52 low, 97 moderate, 50 high
    Black Rot      : 147 low, 38 moderate, 10 high
    Cedar Rust     : 25 low, 107 moderate, 68 high
    Healthy        : 99 kept

This second pass focuses only on the remaining:
    - Black Rot images
    - Cedar Apple Rust images

Target:
    Black Rot  -> at least 50 moderate and 50 high
    Cedar Rust -> at least 50 low

The script reads the existing:
    outputs/severity_selection.csv

and NEVER overwrites previous decisions.

Place this file at:
    D:/23BLC1297/apple_disease_project/src/severity/severity_selector_targeted.py
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
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_FILE = PROJECT_ROOT / "metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "severity_selection.csv"

RANDOM_SEED = 123


# ============================================================
# TARGETS
# ============================================================

TARGETS = {
    "black_rot": {
        "moderate": 50,
        "high": 50,
    },
    "cedar_rust": {
        "low": 50,
    },
}

MAX_IMAGE_W = 760
MAX_IMAGE_H = 650


# ============================================================
# CLASS NORMALIZATION
# ============================================================

def normalize_class_name(value):
    text = str(value).strip().lower()
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
# IMAGE PATH
# ============================================================

def resolve_image_path(image_path_value):
    raw = str(image_path_value).strip()

    if not raw:
        return None

    normalized = raw.replace("/", "\\")
    path = Path(normalized)

    if path.is_absolute() and path.exists():
        return path

    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate

    cleaned = normalized.lstrip(".\\/ ")
    candidate = PROJECT_ROOT / Path(cleaned)

    if candidate.exists():
        return candidate

    return None


# ============================================================
# LOAD EXISTING SELECTIONS
# ============================================================

def load_selection_csv():
    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f"Existing selection file not found:\n{OUTPUT_FILE}\n\n"
            "Run the first selector before running this targeted selector."
        )

    df = pd.read_csv(OUTPUT_FILE)

    required = {
        "image_path",
        "class",
        "internal_class",
        "selection",
        "reviewed",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"severity_selection.csv is missing columns: {sorted(missing)}"
        )

    return df


# ============================================================
# TARGET DEFICITS
# ============================================================

def calculate_deficits(df):
    deficits = {}

    for class_name, targets in TARGETS.items():
        deficits[class_name] = {}

        for severity, target_count in targets.items():
            current_count = int(
                (
                    (df["internal_class"] == class_name)
                    & (df["selection"] == severity)
                ).sum()
            )

            deficits[class_name][severity] = max(
                0,
                target_count - current_count,
            )

    return deficits


def print_status(df, deficits):
    print("\nCurrent selection counts:")
    table = (
        df.groupby(
            ["internal_class", "selection"]
        )
        .size()
        .unstack(fill_value=0)
    )

    print(table.to_string())

    print("\nRemaining targets:")

    for class_name, values in deficits.items():
        print(f"  {class_name}:")
        for severity, deficit in values.items():
            print(
                f"      {severity}: {deficit} more needed"
            )

    print()


# ============================================================
# BUILD TARGETED REVIEW POOL
# ============================================================

def build_targeted_pool(df, deficits):
    """
    Review all remaining Black Rot and Cedar Rust images.

    We deliberately review the complete remaining pool because the
    first-pass labels showed that the desired severity categories are
    relatively rare. We do not want to randomly sample them away.
    """

    rng = random.Random(RANDOM_SEED)

    target_classes = set(TARGETS.keys())

    # Only images from the targeted disease classes.
    subset = df[
        df["internal_class"].isin(target_classes)
    ].copy()

    # Only images that have NOT already received a decision.
    subset = subset[
        subset["selection"].isna()
        | (subset["selection"].astype(str).str.strip() == "")
    ].copy()

    records = subset.to_dict("records")
    rng.shuffle(records)

    candidates = []

    for record in records:
        image_path = resolve_image_path(
            record["image_path"]
        )

        if image_path is None:
            print(
                "WARNING: image file not found:",
                record["image_path"],
            )
            continue

        candidates.append({
            "image_path": str(record["image_path"]),
            "class": str(record["class"]),
            "internal_class": str(record["internal_class"]),
            "selection": "",
            "reviewed": 0,
        })

    return candidates


# ============================================================
# GUI
# ============================================================

class TargetedSelector:

    def __init__(self, root, candidates, deficits):
        self.root = root
        self.remaining = candidates
        self.deficits = deficits

        self.index = 0
        self.history = []
        self.current_photo = None

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.root.title(
            "Apple Disease Severity - Targeted Second Pass"
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
            text="Targeted Severity Review - Round 2",
            font=("Arial", 18, "bold"),
        )
        title.pack()

        self.info_label = tk.Label(
            header,
            text="",
            font=("Arial", 11),
        )
        self.info_label.pack(pady=(4, 0))

        self.target_label = tk.Label(
            header,
            text="",
            font=("Arial", 10),
        )
        self.target_label.pack()

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
    # DISPLAY
    # --------------------------------------------------------

    def show_current(self):
        if self.index >= len(self.remaining):
            self.show_finished()
            return

        item = self.remaining[self.index]
        image_path = resolve_image_path(
            item["image_path"]
        )

        if image_path is None:
            self.index += 1
            self.show_current()
            return

        try:
            image = Image.open(
                image_path
            ).convert("RGB")
        except Exception as exc:
            self.path_label.config(
                text=(
                    f"Could not open image:\n"
                    f"{image_path}\n{exc}"
                )
            )
            self.index += 1
            self.show_current()
            return

        image.thumbnail(
            (MAX_IMAGE_W, MAX_IMAGE_H),
            Image.Resampling.LANCZOS,
        )

        self.current_photo = ImageTk.PhotoImage(
            image
        )

        self.image_label.config(
            image=self.current_photo,
            text="",
        )

        class_name = (
            item["internal_class"]
            .replace("_", " ")
            .title()
        )

        self.info_label.config(
            text=(
                f"Candidate {self.index + 1} / "
                f"{len(self.remaining)}"
                f"     |     Class: {class_name}"
            )
        )

        target_text = []

        for severity, deficit in self.deficits[
            item["internal_class"]
        ].items():
            if deficit > 0:
                target_text.append(
                    f"{severity}: {deficit} needed"
                )

        if target_text:
            self.target_label.config(
                text="TARGETS: " + "   |   ".join(target_text)
            )
        else:
            self.target_label.config(
                text="No remaining target deficit for this class."
            )

        self.path_label.config(
            text=str(image_path)
        )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    def handle_decision(self, decision):
        if self.index >= len(self.remaining):
            return

        item = self.remaining[self.index]
        class_name = item["internal_class"]

        item["selection"] = decision
        item["reviewed"] = 1

        # Record every decision. The CSV is updated immediately.
        self.history.append(
            (self.index, item.copy())
        )

        self.save_row(item)

        # Reduce target deficit when a target category is selected.
        if (
            class_name in self.deficits
            and decision in self.deficits[class_name]
            and self.deficits[class_name][decision] > 0
        ):
            self.deficits[class_name][decision] -= 1

        self.index += 1

        # Once every target is reached, we can stop.
        if self.all_targets_reached():
            self.show_targets_reached()
            return

        self.show_current()

    # --------------------------------------------------------
    # TARGET STATUS
    # --------------------------------------------------------

    def all_targets_reached(self):
        for values in self.deficits.values():
            for deficit in values.values():
                if deficit > 0:
                    return False

        return True

    def show_targets_reached(self):
        self.image_label.config(
            image="",
            text=(
                "TARGETS REACHED\n\n"
                "The requested additional severity candidates "
                "have been selected.\n\n"
                "Press Esc to close."
            ),
            font=("Arial", 18, "bold"),
        )

        self.info_label.config(
            text="Targeted second-pass selection complete."
        )

        self.target_label.config(
            text="All target deficits are now zero."
        )

        self.path_label.config(text="")

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    def save_row(self, item):
        rows = []

        if OUTPUT_FILE.exists():
            try:
                with OUTPUT_FILE.open(
                    "r",
                    newline="",
                    encoding="utf-8-sig",
                ) as file:
                    rows = list(
                        csv.DictReader(file)
                    )
            except Exception:
                rows = []

        found = False

        for row in rows:
            if row.get("image_path") == item["image_path"]:
                row.update({
                    "class": item["class"],
                    "internal_class": item["internal_class"],
                    "selection": item["selection"],
                    "reviewed": "1",
                })
                found = True
                break

        if not found:
            rows.append({
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
            writer.writerows(rows)

    # --------------------------------------------------------
    # UNDO
    # --------------------------------------------------------

    def undo(self):
        if not self.history:
            return

        previous_index, item = self.history.pop()

        self.remove_row(item["image_path"])

        # Restore target deficit if this was a target label.
        class_name = item["internal_class"]
        decision = item["selection"]

        if (
            class_name in TARGETS
            and decision in TARGETS[class_name]
        ):
            # Recalculate from CSV after removal instead of guessing.
            updated = pd.read_csv(OUTPUT_FILE)
            self.deficits = calculate_deficits(
                updated
            )

        self.index = previous_index
        self.show_current()

    def remove_row(self, image_path):
        if not OUTPUT_FILE.exists():
            return

        try:
            with OUTPUT_FILE.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                rows = list(
                    csv.DictReader(file)
                )

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
    # FINISH
    # --------------------------------------------------------

    def show_finished(self):
        self.image_label.config(
            image="",
            text=(
                "Second-pass review complete.\n\n"
                f"Results saved to:\n{OUTPUT_FILE}"
            ),
            font=("Arial", 18, "bold"),
        )

        self.info_label.config(
            text="All remaining targeted images were reviewed."
        )

        self.target_label.config(
            text="Check the terminal for final target status."
        )

        self.path_label.config(text="")

    def close(self):
        self.root.destroy()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("APPLE DISEASE SEVERITY - TARGETED SECOND PASS")
    print("=" * 70)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Selection    : {OUTPUT_FILE}")
    print()

    try:
        df = load_selection_csv()
        deficits = calculate_deficits(df)

        print_status(df, deficits)

        if all(
            deficit == 0
            for values in deficits.values()
            for deficit in values.values()
        ):
            print("All targeted quotas are already satisfied.")
            input("Press Enter to exit...")
            return

        candidates = build_targeted_pool(
            df,
            deficits,
        )

        if not candidates:
            print(
                "No remaining Black Rot/Cedar Rust candidates "
                "are available."
            )
            print(
                "The remaining targets cannot be filled from "
                "the current dataset without changing the plan."
            )
            input("Press Enter to exit...")
            return

        print(
            f"Targeted candidates available: {len(candidates)}"
        )

    except Exception as exc:
        print(f"ERROR: {exc}")
        input("Press Enter to exit...")
        sys.exit(1)

    root = tk.Tk()

    TargetedSelector(
        root,
        candidates,
        deficits,
    )

    root.mainloop()

    # Final terminal report.
    try:
        final_df = pd.read_csv(OUTPUT_FILE)
        final_deficits = calculate_deficits(
            final_df
        )

        print("\nFINAL TARGET STATUS")
        print("-------------------")

        for class_name, values in final_deficits.items():
            for severity, deficit in values.items():
                print(
                    f"{class_name:12s} "
                    f"{severity:10s} "
                    f"remaining deficit = {deficit}"
                )

    except Exception as exc:
        print(
            f"Could not print final status: {exc}"
        )


if __name__ == "__main__":
    main()
