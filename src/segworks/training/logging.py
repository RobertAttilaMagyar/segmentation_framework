import logging
from pathlib import Path


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "train.log"

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    logger = logging.getLogger()  # root logger
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
