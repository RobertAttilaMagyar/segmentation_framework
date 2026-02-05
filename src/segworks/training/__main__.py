import logging
from pathlib import Path
from typing import Any

import torch

from ..config_parsing import build_pipeline
from .trainer import Trainer

logger = logging.getLogger(__name__)


def main(pipeline: dict[str, Any]):
    model = pipeline["model"]
    dataset = pipeline["dataset"]
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
    output_dir = Path(cfg["output_path"]) if cfg and "output_path" in cfg else Path("./out")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "model_final.pt"
    torch.save(trained_model.state_dict(), output_path)

    logger.info(f"Training complete. Model saved to {output_path}")


if __name__ == "__main__":
    import argparse

    from .logging import configure_logging

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration YAML for training.",
    )
    args = parser.parse_args()
    pipeline = build_pipeline(args.config)

    logger.info(f"Starting training based on config file: {args.config}")

    logger = configure_logging(pipeline["output_path"])

    main(pipeline)
