from torchvision import transforms

from ._base import BaseMaskAwareTransform


class MaskAwareToTensor(BaseMaskAwareTransform):
    def __init__(self):
        super().__init__()

        self.transformer = transforms.ToTensor()

    def forward(self, img, mask):
        return self.transformer(img), self.transformer(mask)

    def __call__(self, img, mask):
        return self.forward(img, mask)
