from abc import ABC, abstractmethod


class BaseMaskAwareTransform(ABC):
    @abstractmethod
    def __call__(self, img, mask):
        ...
