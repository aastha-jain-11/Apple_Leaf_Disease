from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ============================================================
# Output directory
# ============================================================

OUTPUT_DIR = Path("outputs") / "visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Ablation study results
# ============================================================

experiments = [
    "C_FULL",
    "C_NO_AUG",
    "C_NO_WEIGHTS",
    "C_NO_SCHEDULER",
    "C_SCRATCH",
]

macro_f1 = [
    1.0000,
    1.0000,
    1.0000,
    1.0000,
    0.9645,
]

best_epoch = [
    2,
    3,
    1,
    3,
    15,
]

pretrained = [True, True, True, True, False]
augmentation = [True, False, True, True, True]
class_weights = [True, True, False, True, True]
scheduler = [True, True, True, False, True]


# ============================================================
# Figure 1: Classification pipeline / model architecture
# ============================================================

def create_model_architecture_figure():

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # --------------------------------------------------------
    # Helper function for boxes
    # --------------------------------------------------------

    def add_box(x, y, width, height, title, text):
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.5,
            edgecolor="black",
            facecolor="white",
        )

        ax.add_patch(box)

        ax.text(
            x + width / 2,
            y + height - 0.45,
            title,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

        ax.text(
            x + width / 2,
            y + height / 2 - 0.15,
            text,
            ha="center",
            va="center",
            fontsize=10.5,
        )

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    add_box(
        0.4, 3.0, 2.0, 2.0,
        "Input",
        "Apple leaf image\n256 × 256 × 3"
    )

    add_box(
        3.0, 3.0, 2.2, 2.0,
        "Preprocessing",
        "Resize / normalization\nTraining augmentation"
    )

    add_box(
        5.9, 2.6, 2.5, 2.8,
        "ConvNeXt V2 Base",
        "Pretrained backbone\nFeature extraction\nFine-tuning"
    )

    add_box(
        9.1, 3.0, 2.0, 2.0,
        "Classifier",
        "Global features\nFully connected layer"
    )

    add_box(
        11.7, 3.0, 1.9, 2.0,
        "Output",
        "4 disease classes\nSoftmax prediction"
    )

    # --------------------------------------------------------
    # Arrows
    # --------------------------------------------------------

    def add_arrow(x1, y1, x2, y2):
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=1.5,
        )
        ax.add_patch(arrow)

    add_arrow(2.4, 4.0, 3.0, 4.0)
    add_arrow(5.2, 4.0, 5.9, 4.0)
    add_arrow(8.4, 4.0, 9.1, 4.0)
    add_arrow(11.1, 4.0, 11.7, 4.0)

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    ax.text(
        7.0,
        7.15,
        "ConvNeXt V2 Base Classification Model",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    ax.text(
        7.0,
        6.55,
        "Complete configuration used in C_FULL",
        ha="center",
        va="center",
        fontsize=11,
    )

    config_text = (
        "Pretrained weights  •  Data augmentation  •  Class-weighted loss  •  "
        "Cosine learning-rate scheduler  •  AMP"
    )

    ax.text(
        7.0,
        1.65,
        config_text,
        ha="center",
        va="center",
        fontsize=10.5,
    )

    ax.text(
        7.0,
        0.85,
        "Training: 2,214 images   |   Validation: 475 images   |   Test: 475 images",
        ha="center",
        va="center",
        fontsize=10.5,
    )

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "classification_model_architecture.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR / "classification_model_architecture.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Figure 2: Ablation Macro F1
# ============================================================

def create_ablation_f1_figure():

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(experiments))

    bars = ax.bar(
        x,
        macro_f1,
        width=0.65,
        edgecolor="black",
        linewidth=1.0,
    )

    # Highlight C_FULL without hardcoding colors
    bars[0].set_hatch("//")

    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=11)

    ax.set_ylabel("Best Validation Macro F1", fontsize=12)

    ax.set_xlabel("Experiment", fontsize=12)

    ax.set_ylim(0.94, 1.005)

    ax.set_title(
        "Ablation Study: Validation Macro F1",
        fontsize=16,
        fontweight="bold",
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    # Add values above bars
    for bar, value in zip(bars, macro_f1):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.0015,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )

    # Annotation
    ax.annotate(
        "Pretrained model configurations",
        xy=(2.0, 0.997),
        xytext=(2.0, 0.948),
        ha="center",
        fontsize=10,
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.2,
        ),
    )

    ax.annotate(
        "Training from scratch\nreduced validation Macro F1",
        xy=(4, 0.9645),
        xytext=(3.45, 0.948),
        ha="center",
        fontsize=10,
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.2,
        ),
    )

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "ablation_macro_f1.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR / "ablation_macro_f1.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Figure 3: Configuration matrix
# ============================================================

def create_ablation_configuration_figure():

    configuration = np.array([
        [1, 1, 1, 1],  # C_FULL
        [1, 0, 1, 1],  # C_NO_AUG
        [1, 1, 0, 1],  # C_NO_WEIGHTS
        [1, 1, 1, 0],  # C_NO_SCHEDULER
        [0, 1, 1, 1],  # C_SCRATCH
    ])

    labels = [
        "Pretrained",
        "Augmentation",
        "Class weights",
        "Scheduler",
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    image = ax.imshow(
        configuration,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)

    ax.set_yticks(np.arange(len(experiments)))
    ax.set_yticklabels(experiments, fontsize=11)

    ax.set_xlabel("Configuration component", fontsize=12)

    ax.set_ylabel("Experiment", fontsize=12)

    ax.set_title(
        "Ablation Study Configuration Matrix",
        fontsize=16,
        fontweight="bold",
    )

    # Add check / dash labels
    for i in range(configuration.shape[0]):
        for j in range(configuration.shape[1]):

            if configuration[i, j] == 1:
                text = "✓"
            else:
                text = "–"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
            )

    # Add F1 to right side
    for i, value in enumerate(macro_f1):

        ax.text(
            len(labels) + 0.25,
            i,
            f"F1 = {value:.4f}",
            va="center",
            fontsize=10.5,
        )

    ax.set_xlim(-0.5, len(labels) + 1.4)

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "ablation_configuration_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR / "ablation_configuration_matrix.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Figure 4: Best epoch comparison
# ============================================================

def create_best_epoch_figure():

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(experiments))

    bars = ax.bar(
        x,
        best_epoch,
        width=0.65,
        edgecolor="black",
        linewidth=1.0,
    )

    bars[0].set_hatch("//")

    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=11)

    ax.set_ylabel("Best Epoch", fontsize=12)

    ax.set_xlabel("Experiment", fontsize=12)

    ax.set_title(
        "Ablation Study: Epoch with Best Validation Macro F1",
        fontsize=15,
        fontweight="bold",
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    for bar, value in zip(bars, best_epoch):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.3,
            str(value),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "ablation_best_epoch.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR / "ablation_best_epoch.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    create_model_architecture_figure()
    create_ablation_f1_figure()
    create_ablation_configuration_figure()
    create_best_epoch_figure()

    print("\nVisualization completed.")
    print(f"Files saved to: {OUTPUT_DIR.resolve()}")

    print("\nGenerated files:")

    for file in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {file.name}")