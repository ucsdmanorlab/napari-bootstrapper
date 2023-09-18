__version__ = "0.0.1"

from .sample_data._load import cremi_sample
from .widgets._model_2d import (
    TrainWidget,
    model_config_widget,
    train_config_widget,
)

__all__ = (
    "cremi_sample",
    "train_config_widget",
    "model_config_widget",
    "TrainWidget",
)
