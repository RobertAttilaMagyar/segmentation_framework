from typing import Literal

from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def export(format: Literal['onnx']):
        ...