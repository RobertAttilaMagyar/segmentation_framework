from enum import Enum

from ._base import BaseMaskAwareTransform
from .crop import MaskAwareRandomCrop
from .resize import MaskAwareResize
from .to_tensor import MaskAwareToTensor


class MaskTransform(Enum):
    RANDOM_CROP = 'random_crop'
    RESIZE = 'resize'

MASK_TRANSFORM_REGISTRY: dict[str, BaseMaskAwareTransform] = {
    MaskTransform.RANDOM_CROP: MaskAwareRandomCrop,
    MaskTransform.RESIZE: MaskAwareResize,
}

def build_transforms(cfg):
    transforms = [MaskAwareToTensor()]

    for item in cfg:
        name, params = next(iter(item.items()))
        transforms.append(MASK_TRANSFORM_REGISTRY[MaskTransform(name)](**params))

    return transforms
