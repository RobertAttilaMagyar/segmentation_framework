from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset


@dataclass
class ImageMaskPair:
    image_path: str | Path
    mask_path: str | Path

    def __post_init__(self):
        if isinstance(self.image_path, str):
            self.image_path = Path(self.image_path)
        if isinstance(self.mask_path, str):
            self.mask_path = Path(self.mask_path)

    @property
    def exists(self) -> bool:
        return self.image_path.exists() and self.mask_path.exists()

    @property
    def values(self) -> tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(self.image_path).astype(np.float32)
        msk = (cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE) != 0).astype(np.float32)
        return img, msk

    def visualize(self):
        img, mask = self.values
        plt.imshow(img)
        plt.imshow(mask, alpha=0.3)
        plt.show()


class SegmentationMaskData(Dataset):
    def __init__(
        self,
        labels_root_path: str | Path,
        images_root_path: str | Path,
        transforms: list[Callable] | None = None,
    ):
        labels_root_path = Path(labels_root_path)
        images_root_path = Path(images_root_path)
        self.path_pairs: list[ImageMaskPair] = []

        # NOTE: Pay extra attention to use appropriate transforms,
        # consider the masks as well in geometric transforms
        self.transforms = transforms or []
        for mask_path in labels_root_path.rglob("*.png"):
            image_path = images_root_path / Path(*mask_path.parts[-3:]).with_suffix(
                ".jpg"
            )
            if (pair := ImageMaskPair(image_path, mask_path)).exists:
                self.path_pairs.append(pair)

    def __getitem__(self, index):
        img, mask = self.path_pairs[index].values

        for transform in self.transforms:
            img, mask = transform(img, mask)

        return img, mask.squeeze()

    def __len__(self):
        return len(self.path_pairs)
