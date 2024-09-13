from typing import Union

from omegaconf import DictConfig, ListConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.logger import Logger


def get_logger(config: Union[DictConfig, ListConfig]) -> Logger:
    if hasattr(config, "logger"):
        return TensorBoardLogger(save_dir=config.logger.experiment, name=config.logger.name)
    return TensorBoardLogger("lightning_logs/", name="test_training")
