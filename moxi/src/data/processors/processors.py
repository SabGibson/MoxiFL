from typing import Tuple, Union
import torch
import numpy as np
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from .base import PreprocessingStrategy


class ImageClassificationPreprocessing(PreprocessingStrategy):
    """
    Preprocessing strategy for image classification tasks.

    NOTE:
    For federated CIFAR-10, preprocessing is handled in `collate_fn` in core.py.
    This class is useful if you work with non-federated image datasets.
    """

    def __init__(self, image_size: int = 32):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def preprocess(
        self,
        features: Union[np.ndarray, torch.Tensor, dict],
        labels: Union[np.ndarray, torch.Tensor, dict],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Features
        if isinstance(features, dict):
            images = features["img"]
            if isinstance(images, list):
                images = np.array(images)
        else:
            images = features

        if isinstance(images, torch.Tensor) and len(images.shape) == 4:
            transformed_features = images
        else:
            if len(images.shape) == 3:
                images = images.reshape(-1, 3, images.shape[1], images.shape[2])
            transformed_features = torch.stack([self.transform(img) for img in images])

        # Labels
        if isinstance(labels, dict):
            labels = labels.get("label")
        if isinstance(labels, torch.Tensor):
            labels_tensor = labels.detach().clone().to(dtype=torch.long)
        else:
            labels_tensor = torch.tensor(labels, dtype=torch.long)

        return transformed_features, labels_tensor


class BinaryClassificationPreprocessing(PreprocessingStrategy):
    """Preprocessing strategy for binary classification tasks."""

    def __init__(self):
        self.scaler = StandardScaler()

    def preprocess(
        self, features: np.ndarray, labels: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scaled_features = self.scaler.fit_transform(features)
        features_tensor = torch.tensor(scaled_features, dtype=torch.float)

        if isinstance(labels, torch.Tensor):
            labels_tensor = labels.detach().clone().to(dtype=torch.float)
        else:
            labels_tensor = torch.tensor(labels, dtype=torch.float)

        return features_tensor, labels_tensor


class RegressionPreprocessing(PreprocessingStrategy):
    """Preprocessing strategy for regression tasks."""

    def __init__(self):
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()  # <-- scale the targets

    def preprocess(
        self, features: np.ndarray, labels: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Scale features
        X_scaled = self.X_scaler.fit_transform(features)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Scale labels to prevent exploding gradients
        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy().reshape(-1, 1)
        else:
            labels_np = np.array(labels).reshape(-1, 1)

        y_scaled = self.y_scaler.fit_transform(labels_np)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

        return X_tensor, y_tensor

    def inverse_transform_labels(self, y_tensor: torch.Tensor) -> np.ndarray:
        """Convert scaled predictions back to original scale."""
        return self.y_scaler.inverse_transform(y_tensor.detach().cpu().numpy())
