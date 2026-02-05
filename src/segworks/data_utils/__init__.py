from .dataset import SegmentationMaskData

__all__ = ["SegmentationMaskData"]


def build_dataset(cfg, transforms=None):
    images_root = cfg["images_root_dir"]
    labels_root = cfg["labels_root_dir"]

    return SegmentationMaskData(
        images_root_path=images_root,
        labels_root_path=labels_root,
        transforms=transforms,
    )
