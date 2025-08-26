from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import threading
import time
import uuid
import logging
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeType(Enum):
    CLIENT = "client"
    ORCHESTRATOR = "orchestrator"


class FLMode(Enum):
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"


@dataclass
class ModelUpdate:
    """Represents a model update with metadata"""

    node_id: str
    model_weights: Dict[str, Any]
    metrics: Dict[str, float]
    round_number: int
    timestamp: float
    data_size: int


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""

    epochs: int
    batch_size: int
    learning_rate: float
    local_rounds: int
    min_participants: int
    max_participants: Optional[int] = None
    timeout: float = 300.0  # seconds


class AbstractMoxiNode(ABC):
    """Abstract base class for all nodes in the federated learning network"""

    def __init__(self, node_id: str, node_type: NodeType, parent_network: Any):
        self.parent_network: any = parent_network
        self.node_id = node_id
        self.trainer = None
        self._node_type = node_type
        self.is_active = False
        self.current_round = 0
        self.model_version = 0
        self.config = None

    @property
    def neighbours(self) -> List[str]:
        """List of neighboring node IDs"""
        return self.parent_network.graph.neighbors(self.node_id)

    @property
    def nodetype(self) -> str:
        """Return the type of this node"""
        return self._node_type.value

    @abstractmethod
    def initialize(self, model, data, config: TrainingConfig) -> bool:
        """Initialize the node with training configuration"""

        pass

    @abstractmethod
    def receive_model_update(self, update: ModelUpdate) -> bool:
        """Receive a model update from another node"""
        pass

    @abstractmethod
    def send_model_update(self, target_node_id: str, update: ModelUpdate) -> bool:
        """Send a model update to another node"""
        pass

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the node"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "is_active": self.is_active,
            "current_round": self.current_round,
            "model_version": self.model_version,
            "neighbors": self.neighbors,
        }

    def __repr__(self):
        return self.get_status().__repr__()
