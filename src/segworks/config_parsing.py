from dataclasses import fields
from pathlib import Path

import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data_utils import build_dataset
from .models import build_model
from .training import TrainingArgs
from .transforms import build_transforms

OPTIMIZERS = {
    "Adam": optim.Adam,
}

SCHEDULERS = {
    "CosineAnnealingScheduler": CosineAnnealingLR,
}

def load_config(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def build_optimizer(model, cfg):
    return OPTIMIZERS[cfg["optimizer"]](
        model.parameters(),
        lr=cfg["learning_rate"],
    )

def build_scheduler(optimizer, cfg):
    return SCHEDULERS[cfg["scheduler"]](
        optimizer,
        T_max=cfg["epochs"],
    )

def build_training_args(cfg) -> TrainingArgs:
    valid_fields = {f.name for f in fields(TrainingArgs)}
    kwargs = {k: v for k, v in cfg.items() if k in valid_fields}
    return TrainingArgs(**kwargs)


def build_pipeline(config_path):
    cfg = load_config(config_path)

    transforms = build_transforms(cfg["transformations"]["training"])
    dataset = build_dataset(cfg["dataset"]["training"], transforms)

    model = build_model(cfg["model"])

    training_args = build_training_args(cfg["training_params"])

    optimizer = build_optimizer(model, cfg["training_params"])
    scheduler = build_scheduler(optimizer, cfg["training_params"])
    output_path = Path(cfg.get("output_path", r".\out"))

    return {
        "model": model,
        "dataset": dataset,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "training_args": training_args,
        "output_path": output_path,
        "validation_fraction": cfg["training_params"].get("validation_fraction", None)
    }

if __name__ == "__main__":
    config_file_path = r'.\initial_training_config.yml'

    print(build_pipeline(config_file_path))


