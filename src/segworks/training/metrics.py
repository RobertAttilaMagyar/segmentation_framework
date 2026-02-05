from enum import Enum

import torch
from torch.utils.data import DataLoader, Dataset


class EvaluationMetric(Enum):
    DICE = "dice"
    IOU = "IoU"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"


# Currently only for binary segmentation
class Evaluator:
    def __init__(
        self,
        validation_dataset: Dataset,
        *,
        num_classes: int = 2,
        num_workers: int | None = None,
    ):
        if num_classes != 2:
            raise NotImplementedError(
                "Current implementation only supports binary segmentation"
            )
        self.loader = DataLoader(
            dataset=validation_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=num_workers or 1,
            pin_memory=True,
        )
        self.confusion_matrix = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64
        )  # [[TP, FN], [FP, TN]]

        self.evaluated = False

    def reset(self):
        self.confusion_matrix = torch.zeros_like(self.confusion_matrix, torch.int64)
        self.evaluated = False

    def _process_outputs(self, output: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(output) > 0.5).long()

    @property
    def dice_score(self):
        assert self.evaluated, (
            "Performance metric can only be accessed after evaluation"
        )
        return (
            2
            * self.confusion_matrix[0, 0]
            / (
                torch.sum(self.confusion_matrix[:, 0])
                + torch.sum(self.confusion_matrix[0, :])
            )
        ).item() or 0

    @property
    def precision(self):
        assert self.evaluated, (
            "Performance metric can only be accessed after evaluation"
        )
        return (
            self.confusion_matrix[0, 0] / torch.sum(self.confusion_matrix[:, 0])
        ).item() or 0

    @property
    def recall(self):
        assert self.evaluated, (
            "Performance metric can only be accessed after evaluation"
        )
        return (
            self.confusion_matrix[0, 0] / torch.sum(self.confusion_matrix[0, :])
        ).item() or 0

    @property
    def f1_score(self):
        assert self.evaluated, (
            "Performance metric can only be accessed after evaluation"
        )
        return (
            2 * (self.precision * self.recall) / (self.precision + self.recall)
        ) or 0

    @property
    def iou(self) -> float:
        assert self.evaluated, (
            "Performance metric can only be accessed after evaluation"
        )
        return (
            self.confusion_matrix[0, 0]
            / (
                self.confusion_matrix[0, 0]
                + self.confusion_matrix[0, 1]
                + self.confusion_matrix[1, 0]
            )
        ) or 0

    @staticmethod
    def _calc_confusion_matrix(
        output: torch.Tensor, gt_mask: torch.Tensor
    ) -> torch.Tensor:
        output = output.flatten()
        gt_mask = gt_mask.flatten()
        TP = torch.sum((output == 1) & (gt_mask == 1))
        TN = torch.sum((output == 0) & (gt_mask == 0))
        FP = torch.sum((output == 1) & (gt_mask == 0))
        FN = torch.sum((output == 0) & (gt_mask == 1))

        return torch.tensor([[TP, FN], [FP, TN]]).to(torch.device("cpu"))

    def _update_metrics(self, output: torch.Tensor, gt_mask: torch.Tensor):
        self.confusion_matrix += self._calc_confusion_matrix(output, gt_mask)

    @torch.no_grad()
    def evaluate_torch(self, model: torch.nn.Module):
        model.eval()
        device = next(iter(model.parameters())).device

        for data, gt_mask in self.loader:
            data = data.to(device)
            gt_mask = gt_mask.to(device)
            output = model(data)
            output = self._process_outputs(output)

            self._update_metrics(output, gt_mask)

        self.evaluated = True

        return self.to_dict()

    def __call__(self, model: torch.nn.Module) -> dict[EvaluationMetric, float]:
        self.reset()

        if isinstance(model, torch.nn.Module):
            self.evaluate_torch(model)

        # TODO: Implement at least for ONNX model format as well
        else:
            raise NotImplementedError(
                f"Evaluating model of type {type(model).__name__} is not supported"
            )

        return self.to_dict()

    def to_dict(self) -> dict[EvaluationMetric, float]:
        assert self.evaluated, (
            "Performance metrics can only be accessed after evaluation"
        )

        return {
            EvaluationMetric.DICE: self.dice_score,
            EvaluationMetric.PRECISION: self.precision,
            EvaluationMetric.RECALL: self.recall,
            EvaluationMetric.F1: self.f1_score,
            EvaluationMetric.IOU: self.iou,
        }
