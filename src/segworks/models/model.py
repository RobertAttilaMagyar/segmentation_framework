from typing import Literal

from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def export(format: Literal["onnx"]):
        match format:
            case "onnx":


                
                raise NotImplementedError()
            case _:
                raise ValueError(f"Unsupported export format {format}")
