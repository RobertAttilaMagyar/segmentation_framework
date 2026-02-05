import torchvision.transforms.functional as F
from torchvision import transforms

from ._base import BaseMaskAwareTransform


class MaskAwareRandomCrop(BaseMaskAwareTransform, transforms.RandomCrop):
    def __init__(
        self,
        *,
        size,
        padding=None,
        pad_if_needed=False,
        fill=0,
        padding_mode="constant",
    ):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)

    def forward(self, img, mask):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            mask = F.pad(mask, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)

        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            mask = F.pad(mask, padding, self.fill, self.padding_mode)

        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            mask = F.pad(mask, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(mask, i, j, h, w)

    def __call__(self, img, mask):
        return self.forward(img, mask)
