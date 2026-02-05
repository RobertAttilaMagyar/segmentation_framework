from enum import Enum
from typing import Any

from torch import nn

from .unet import UNet

__all__ = ["UNet"]


class ModelType(Enum):
    UNET = "UNet"


MODELS_REGISTRY: dict[ModelType, nn.Module] = {
    ModelType.UNET: UNet,
}


def build_model(cfg: dict[str, Any]) -> nn.Module:
    model_type = ModelType(cfg["type"])
    parameters = cfg.get("args", {})
    return MODELS_REGISTRY[model_type](**parameters)
