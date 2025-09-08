from typing import TypedDict, Literal, Optional
from enum import Enum
from moxi.src.common.interfaces.network import MoxiNetworkProtocol


class MoxiNetworkType(Enum):
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"


class MoxiMLModelType(Enum):
    PARAMETRIC = "parametric"


class MoxiNetworkConfiguration(TypedDict):
    network_name: str
    network_type: MoxiNetworkType
    model_type: MoxiMLModelType
    number_rounds: int
    metrics: list[Literal["convergence", "mean_perfomance"]]
    logger: Literal[
        "mlflow",
        "none",
    ]
    network_size: Optional[int]
    adjcency_matrix: Optional[dict[object, list[int]]]
    prebuilt_config: Optional[Literal["star-n", "random_fc"]]
    model: Optional[object]


class MoxiNodeConfig(TypedDict):
    network_ref: MoxiNetworkProtocol
    device_id: str


class NetworkArchitecture(TypedDict):
    nsm: dict[str, object]
