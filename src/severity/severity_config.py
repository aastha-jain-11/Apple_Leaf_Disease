from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEVERITY_ROOT = PROJECT_ROOT / "outputs" / "severity"
ANNOTATION_IMAGES = SEVERITY_ROOT / "annotation_images"
LABELS_JSON = SEVERITY_ROOT / "labels_json"
MASKS = SEVERITY_ROOT / "masks"
SPLIT_CSV = SEVERITY_ROOT / "severity_segmentation_split.csv"
MODEL_ROOT = PROJECT_ROOT / "models" / "segformer" / "severity"
OUTPUT_ROOT = SEVERITY_ROOT / "segformer"

CLASS_NAMES = ["background", "leaf", "lesion"]
NUM_CLASSES = 3
IMAGE_SIZE = 224
SEED = 42

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

SEGFORMER_CHECKPOINT = "nvidia/mit-b0"

BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 30
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 7
