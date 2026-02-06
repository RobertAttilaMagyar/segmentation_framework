from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
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
        msk = (cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE) != 0).astype(np.uint8)
        return img, msk

    def visualize(self, predicted_mask: Image.Image | None = None, opacity: int = 100):
        """
        Visualize the image with optional ground-truth and predicted mask overlays.

        Parameters
        ----------
        predicted_mask : PIL.Image or None
            Optional predicted mask to overlay in red.
        opacity : int
            Opacity of overlays (0–255).
        """

        # Load image + mask as PIL
        img = Image.open(self.image_path).convert("RGBA")
        mask = Image.open(self.mask_path).convert("L")

        result = img

        # --- Ground truth mask (green) ---
        if mask is not None:
            mask_np = np.array(mask, dtype=np.uint8)
            mask_np = (mask_np.astype(float) * (opacity / 255)).astype(np.uint8)
            mask_scaled = Image.fromarray(mask_np, mode="L")

            overlay_gt = Image.new("RGBA", img.size, (0, 255, 0, 255))
            overlay_gt.putalpha(mask_scaled)

            result = Image.alpha_composite(result, overlay_gt)

        # --- Predicted mask (red) ---
        if predicted_mask is not None:
            pred_l = predicted_mask.convert("L")
            pred_np = np.array(pred_l, dtype=np.uint8)
            pred_np = (pred_np.astype(float) * (opacity / 255)).astype(np.uint8)
            pred_scaled = Image.fromarray(pred_np, mode="L")

            overlay_pred = Image.new("RGBA", img.size, (255, 0, 0, 255))
            overlay_pred.putalpha(pred_scaled)

            result = Image.alpha_composite(result, overlay_pred)

        # --- Plot ---
        plt.figure(figsize=(8, 8))
        plt.imshow(result)
        plt.axis("off")
        plt.show()


_DATASET_REGISTRY = {}


def register_dataset(name: str):
    def decorator(cls):
        _DATASET_REGISTRY[name] = cls
        return cls

    return decorator


class SegmentationDataset(Dataset):
    def __init__(self, transforms: list[Callable] | None = None):
        super().__init__()
        self.path_pairs: list[ImageMaskPair] = []
        self.transforms = transforms or []

    @abstractmethod
    def obtain_pairs(self) -> None: ...

    def __getitem__(self, index):
        img, mask = self.path_pairs[index].values

        for transform in self.transforms:
            img, mask = transform(img, mask)

        return img, mask.squeeze()

    def __len__(self):
        return len(self.path_pairs)


@register_dataset("folder")
class FolderDataset(SegmentationDataset):
    def __init__(
        self,
        labels_root_path: str | Path,
        images_root_path: str | Path,
        transforms: list[Callable] | None = None,
    ):
        super().__init__(transforms)
        self.labels_root_path = Path(labels_root_path)
        self.images_root_path = Path(images_root_path)
        self.obtain_pairs()

    def obtain_pairs(self):
        for mask_path in self.labels_root_path.rglob("*.png"):
            image_path = self.images_root_path / mask_path.name
            if (pair := ImageMaskPair(image_path, mask_path)).exists:
                self.path_pairs.append(pair)


@register_dataset("lane_detection")
class LaneDetectionDataset(SegmentationDataset):
    def __init__(
        self,
        labels_root_path: str | Path,
        images_root_path: str | Path,
        transforms: list[Callable] | None = None,
    ):
        super().__init__(transforms)
        self.labels_root_path = Path(labels_root_path)
        self.images_root_path = Path(images_root_path)
        self.obtain_pairs()

    def obtain_pairs(self):
        for mask_path in self.labels_root_path.rglob("*.png"):
            image_path = self.images_root_path / Path(
                *mask_path.parts[-3:]
            ).with_suffix(".jpg")
            if (pair := ImageMaskPair(image_path, mask_path)).exists:
                self.path_pairs.append(pair)
