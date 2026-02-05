import re
from pathlib import Path
from typing import Literal

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _validate_resolution_format(resolution: str) -> None:
        pattern = r"^\d+(x\d+)+$"
        if not re.fullmatch(pattern, resolution):
            raise ValueError(
                "Invalid resolution format. Expected format like '1x3x480x480'"
            )

    def _export_to_onnx(
        self,
        file: str | Path,
        *,
        resolution: str,
        **kwargs,
    ) -> None:
        if not isinstance(resolution, str):
            raise TypeError(
                f"Resolution must be a string, got {type(resolution).__name__}"
            )

        self._validate_resolution_format(resolution)

        shape = [int(num) for num in resolution.split("x")]
        if len(shape) != 4:
            raise ValueError(
                f"Resolution must have 4 dimensions (N,C,H,W), got {shape}"
            )

        self.eval()
        device = next(self.parameters()).device
        dummy_input = torch.randn(*shape, device=device)

        torch.onnx.export(
            self,
            dummy_input,
            file,
            **kwargs,
        )

    def export(
        self,
        file: Path | str,
        format: Literal["onnx"],
        **kwargs,
    ) -> None:
        match format:
            case "onnx":
                defaults = {
                    "resolution": "1x3x480x480",
                    "export_params": True,
                    "opset_version": 17,
                    "do_constant_folding": True,
                }
                defaults |= kwargs
                self._export_to_onnx(file, **defaults)

            case _:
                raise ValueError(f"Unsupported export format {format}")
