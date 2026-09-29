from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from severity_config import SEVERITY_ROOT, IMAGE_SIZE

class SeveritySegmentationDataset(Dataset):
    def __init__(self, split_csv, split, training=False):
        self.data = pd.read_csv(split_csv)
        self.data = self.data[self.data["split"] == split].reset_index(drop=True)
        self.training = training
        if len(self.data) == 0:
            raise RuntimeError(f"No samples for split={split}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data.iloc[i]
        cls = row["class"]
        stem = Path(row["filename"]).stem
        mask_path = SEVERITY_ROOT / row["mask_path"]

        image = None
        for ext in [".jpg",".JPG",".jpeg",".JPEG",".png",".PNG"]:
            p = SEVERITY_ROOT / "annotation_images" / cls / f"{stem}{ext}"
            if p.exists():
                image = p
                break
        if image is None:
            raise FileNotFoundError(f"Image missing for {mask_path}")

        image = Image.open(image).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)

        if self.training:
            if torch.rand(1).item() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if torch.rand(1).item() < 0.2:
                image, mask = TF.vflip(image), TF.vflip(mask)

        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.485,0.456,0.406], [0.229,0.224,0.225])
        mask = torch.from_numpy(np.asarray(mask, dtype=np.int64))

        return {"pixel_values": image, "labels": mask,
                "class_name": cls, "filename": row["filename"]}
