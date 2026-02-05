from dataclasses import dataclass

import torch

from .metrics import EvaluationMetric


@dataclass
class TrainingArgs:
    epochs: int = 100
    nominal_batch_size: int = 64
    batch_size: int = 16
    device: torch.device | str | None = None
    early_stopping: int | None = None
    num_workers: int = 1
    validation_frequency: int = 1
    no_validation: bool = False
    selection_score: EvaluationMetric | str = EvaluationMetric.DICE
    seed: int = 42

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(self.device)

        assert self.nominal_batch_size % self.batch_size == 0, (
            f"Invalid batch size ({self.batch_size} for nominal batch size {self.nominal_batch_size})"  # noqa: E501
        )

        if isinstance(self.selection_score, str):
            self.selection_score = EvaluationMetric(self.selection_score)
