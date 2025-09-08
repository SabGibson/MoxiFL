from abc import ABC, abstractmethod
from typing import Tuple, Iterable, Dict, Any
import numpy as np
import torch


class PreprocessingStrategy(ABC):
    """Abstract base class for dataset-specific preprocessing strategies."""

    @abstractmethod
    def preprocess(
        self, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess features and labels according to task requirements."""
        pass
