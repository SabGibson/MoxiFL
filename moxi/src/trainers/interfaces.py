from typing import Protocol, TypedDict, NotRequired, Literal, Callable, Any
from enum import Enum


class MoxiDataset(TypedDict):
    train_data: any
    train_labels: any
    val_data: NotRequired[any]
    val_labels: NotRequired[any]


class MoxiTorchDataset(TypedDict):
    train_data: any
    val_data: NotRequired[any]


class MoxiTrainer(Protocol):
    def train(self) -> None: ...

    def evaluate(self) -> float: ...

    def get_model_weights(self) -> dict: ...

    def set_model_weights(self, weights: dict) -> None: ...


class MoxiTrainerFrameworkType(Enum):
    PYTORCH = "pytorch"
    SCIKIT_LEARN = "sklearn"


class MoxiTrainerConfig(TypedDict):
    framework: MoxiTrainerFrameworkType
    seed: NotRequired[int]
    dtype: NotRequired[Literal["float32", "float64"]]
    # sklearn
    n_estimators: NotRequired[int]
    max_depth: NotRequired[int]

    # pytorch
    learning_rate: NotRequired[float]
    batch_size: NotRequired[int]
    criterion: NotRequired[Callable]
    optimizer: NotRequired[dict[str, Callable]]

    # tensorflow
    epochs: NotRequired[int]
    activation: NotRequired[Literal["relu", "sigmoid", "tanh"]]

    # Metrics
    metrics: NotRequired[dict[str, Callable[[Any, Any], float]]]
