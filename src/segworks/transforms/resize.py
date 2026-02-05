from torchvision import transforms

from ._base import BaseMaskAwareTransform


class MaskAwareResize(BaseMaskAwareTransform, transforms.Resize):
    def __init__(
        self,
        *,
        size,
        interpolation=transforms.InterpolationMode.BILINEAR,
        max_size=None,
        antialias=True,
    ):
        super().__init__(size, interpolation, max_size, antialias)

    def forward(self, img, mask):
        return super().forward(img), super().forward(mask)

    def __call__(self, img, mask):
        return self.forward(img, mask)
