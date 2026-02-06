import logging
from pathlib import Path
from typing import Any

import torch

from .trainer import Trainer

logger = logging.getLogger(__name__)


def train(pipeline: dict[str, Any]):
    model = pipeline["model"]
    dataset = pipeline["training_dataset"]
    optimizer = pipeline["optimizer"]
    scheduler = pipeline["scheduler"]
    training_args = pipeline["training_args"]
    output_path = pipeline["output_path"]
    validation_fraction = pipeline["validation_fraction"]

    # Validation intentionally optional
    validation_dataset = None

    trainer = Trainer(
        training_params=training_args,
        training_dataset=dataset,
        validation_dataset=validation_dataset,
        validation_fraction=validation_fraction,
    )

    logger.info("Starting training")
    trained_model = trainer.train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Output handling
    cfg = pipeline.get("raw_config", None)
    output_dir = (
        Path(cfg["output_path"]) if cfg and "output_path" in cfg else Path("./out")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "model_final.pt"
    torch.save(trained_model.state_dict(), output_path)

    logger.info(f"Training complete. Model saved to {output_path}")
