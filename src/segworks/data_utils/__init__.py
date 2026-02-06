from .dataset import (
    _DATASET_REGISTRY,
    FolderDataset,
    ImageMaskPair,
    LaneDetectionDataset,
    SegmentationDataset,
    register_dataset,
)

__all__ = [
    "SegmentationDataset",
    "ImageMaskPair",
    "LaneDetectionDataset",
    "FolderDataset",
    "register_dataset",
]


def build_dataset(cfg, transforms=None) -> SegmentationDataset:
    dataset_type = cfg.get("type", "folder")  # default
    params = cfg.get("params", {})

    if dataset_type not in _DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset type '{dataset_type}'")

    cls = _DATASET_REGISTRY[dataset_type]
    return cls(transforms=transforms, **params)
