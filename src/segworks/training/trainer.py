import logging
import tempfile
from collections.abc import Callable
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader, Dataset, random_split

from .metrics import Evaluator
from .training_params import TrainingArgs

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        training_params: TrainingArgs,
        training_dataset: Dataset,
        validation_dataset: Dataset | None = None,
        validation_fraction: float | None = None,
    ):
        logger.info(f"Number of training images: {len(training_dataset)}")
        generator = torch.Generator().manual_seed(training_params.seed)
        if validation_fraction:
            assert validation_fraction < 1, (
                f"Fraction should be less then one. Provided: {validation_dataset}"
            )
            assert validation_dataset is None, (
                "Cannot provide validation fraction and validation dataset simultaniously"
            )
            validation_dataset, training_dataset = random_split(
                training_dataset,

                
                [validation_fraction, 1 - validation_fraction],
                generator=generator,
            )
        self.train_loader = DataLoader(
            dataset=training_dataset,
            batch_size=training_params.batch_size,
            shuffle=True,
            num_workers=training_params.num_workers,
            pin_memory=True,
        )

        self.epochs = training_params.epochs
        self.device = training_params.device
        self.early_stopping = training_params.early_stopping
        self.skip_validation = training_params.no_validation

        self.validation_frequency = training_params.validation_frequency

        self.loss_function = self._init_loss_function(training_params)

        if validation_dataset is not None:
            logger.info(f"Number of validation images: {len(validation_dataset)}")
            self.evaluator = Evaluator(validation_dataset, num_workers=training_params.num_workers)

        # For fp16 training
        self.use_amp = training_params.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        self.gradient_accumulation_steps = max(
            training_params.nominal_batch_size // training_params.batch_size, 1
        )

        self.model_selection_score = training_params.selection_score

    def _init_loss_function(
        self,
        training_args: TrainingArgs,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor | float]:
        # At least for now I will only implement weighted BCELoss

        # Also calibrate it only for one batch
        _, masks = next(iter(self.train_loader))
        P = torch.clamp((masks == 1).sum(), min=1)
        N = torch.clamp((masks == 0).sum(), min=1)

        pos_weight = torch.clamp(N / P, max=20.0).to(self.device)

        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def train(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> torch.nn.Module:
        model.to(self.device)

        logger.info(f"Using device: {self.device}")

        best_validation_score = -float("inf")
        last_improvement_epoch = 0

        with tempfile.TemporaryDirectory() as workdir:
            workdir = Path(workdir)
            best_weights_path = workdir / "weights.pt"

            torch.save(model.state_dict(), best_weights_path)

            for epoch in range(self.epochs):
                logger.info(f"Starting epoch {epoch + 1}/{self.epochs}")

                model.train()
                running_loss = 0.0

                optimizer.zero_grad()
                for iteration, (images, masks) in enumerate(tqdm.tqdm(self.train_loader), start=1):
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    with torch.amp.autocast(enabled=self.use_amp):
                        outputs = model(images).squeeze(1)
                        loss = self.loss_function(outputs, masks.float())

                        loss = loss / self.gradient_accumulation_steps

                    self.scaler.scale(loss).backward()

                    if iteration % self.gradient_accumulation_steps == 0:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                    running_loss += loss.item()

                # flush leftover gradients
                if iteration % self.gradient_accumulation_steps == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                scheduler.step()

                logger.info(
                    f"Epoch {epoch + 1} training loss: {running_loss / len(self.train_loader):.4f}"
                )

                if self.skip_validation:
                    logger.info("Skipping validation")
                    continue

                if (epoch + 1) % self.validation_frequency != 0:
                    continue

                with torch.no_grad():
                    validation_score = (validation_dict := self.evaluator(model))[
                        self.model_selection_score
                    ]

                logger.info(f"Validation score: {validation_score:.4f}")
                logger.info(f"Validation metrics: {validation_dict}")

                if validation_score > best_validation_score:
                    logger.info(f"New best model found ({validation_score:.4f})")
                    best_validation_score = validation_score
                    last_improvement_epoch = epoch
                    torch.save(model.state_dict(), best_weights_path)

                if self.early_stopping is not None:
                    epoch_diff = epoch - last_improvement_epoch
                    if epoch_diff > self.early_stopping:
                        logger.info(
                            f"Early stopping triggered (no improvement for {epoch_diff} epochs)"
                        )
                        break

            logger.info("Restoring best model weights")
            model.load_state_dict(torch.load(best_weights_path, map_location=self.device))

        return model
