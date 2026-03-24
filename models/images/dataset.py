import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]


def get_transforms(partition: int) -> transforms.Compose:
    """Train partition (0) gets augmentation, val/test get only normalization."""
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMGNET_MEAN, IMGNET_STD),
        ]
    )
    if partition == 0:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                normalize,
            ]
        )
    return normalize


class CelebASmilingDataset(Dataset):
    def __init__(
        self,
        df_attributes: pd.DataFrame,
        df_eval_partitions: pd.DataFrame,
        img_dir: str,
        partition: int = 0,
        transform: transforms.Compose = None,
    ):
        merged = df_attributes[["image_id", "Smiling"]].merge(
            df_eval_partitions, on="image_id"
        )
        self.data = merged[merged["partition"] == partition].reset_index(drop=True)
        self.attributes = df_attributes[
            df_attributes["image_id"].isin(self.data["image_id"])
        ]
        self.img_dir = img_dir
        self.transform = (
            transform if transform is not None else get_transforms(partition)
        )

        # Pre-compute image ids once — avoids repeated string slicing in __getitem__
        self.image_ids = (
            self.data["image_id"].str.removesuffix(".jpg").astype(int).to_numpy()
        )
        # Pre-compute labels as a tensor — no per-sample conversion overhead
        self.labels = torch.tensor(self.data["Smiling"].to_numpy(), dtype=torch.float32)
        # Note: labels are in {-1, 1} in the original dataset; we can convert to {0, 1} if desired:
        self.labels = (self.labels + 1) / 2

    def __len__(self) -> int:
        return len(self.data)

    def __load_image(self, image_id: int) -> np.ndarray:
        img_path = os.path.join(self.img_dir, f"{image_id:06d}.jpg")
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.__load_image(self.image_ids[idx])
        label = self.labels[idx]
        return self.transform(image), label
