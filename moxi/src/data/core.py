import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from typing import Tuple, Iterable, Dict, Any, List, Optional
import datasets
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner
from .processors import (
    BinaryClassificationPreprocessing,
    RegressionPreprocessing,
)
import logging

# Configure logging
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ------------------------------
# Collate function for CIFAR-10
# ------------------------------


def collate_fn(batch):
    """Collate function for federated CIFAR-10."""
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32), antialias=True),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    images, labels = [], []
    for item in batch:
        image = item["img"]  # <-- CIFAR10 dataset column
        label = item["label"]
        image = transform(image)
        images.append(image)
        labels.append(label)

    return torch.stack(images), torch.tensor(labels)


# ------------------------------
# Limited IterableDataset
# ------------------------------


class LimitedIterableDataset(IterableDataset):
    """Wraps an iterable dataset and limits the number of samples yielded."""

    def __init__(
        self, base_dataset: IterableDataset, max_samples: Optional[int] = None
    ):
        self.base_dataset = base_dataset
        self.max_samples = max_samples

    def __iter__(self):
        count = 0
        for item in self.base_dataset:
            if self.max_samples is not None and count >= self.max_samples:
                break
            yield item
            count += 1


# ------------------------------
# Helpers
# ------------------------------


def create_dataloader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TensorDataset(features, labels)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def extract_features_labels(ds, task: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert HuggingFace dataset to tensors depending on task."""
    if task == "binary_classification":
        labels = torch.tensor([1 if d == "M" else 0 for d in ds["diagnosis"]])
        feature_cols = [
            c for c in ds.column_names if c not in ["id", "diagnosis", "Unnamed: 32"]
        ]
        features = torch.tensor(
            [[float(row[c]) for c in feature_cols] for row in ds]
        ).float()

    elif task == "regression":
        numeric_cols = [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ]
        features, labels = [], []
        for row in ds:
            try:
                features.append([float(row[c]) for c in numeric_cols[:-1]])
                labels.append(float(row[numeric_cols[-1]]))
            except (KeyError, ValueError, TypeError):
                continue
        features = torch.tensor(features).float()
        labels = torch.tensor(labels).float()

    else:
        raise ValueError(f"Unsupported task for feature extraction: {task}")

    return features, labels


# ------------------------------
# Federated CIFAR-10 loaders
# ------------------------------


def create_federated_cifar10_loaders(
    dataset_name: str,
    partitioner: Partitioner,
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int] = None,
) -> Tuple[List[DataLoader], DataLoader]:
    """Create federated CIFAR-10 DataLoaders using Flower Datasets."""

    print(f"Loading federated CIFAR-10 dataset: {dataset_name}")

    fds = FederatedDataset(dataset=dataset_name, partitioners={"train": partitioner})
    fds.load_partition(0, "train")
    train_loaders = []

    for client_id in range(partitioner.num_partitions):
        partition = fds.load_partition(client_id, "train")
        iterable_dataset = partition.to_iterable_dataset()
        limited_dataset = LimitedIterableDataset(iterable_dataset, max_samples)

        client_loader = DataLoader(
            limited_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=0,
        )
        train_loaders.append(client_loader)

    # Test set (global, not partitioned)
    original_dataset = datasets.load_dataset(dataset_name)
    if "test" in original_dataset:
        test_dataset = original_dataset["test"]
    else:
        train_test_split = original_dataset["train"].train_test_split(test_size=0.2)
        test_dataset = train_test_split["test"]

    test_iterable = test_dataset.to_iterable_dataset()
    limited_test_dataset = LimitedIterableDataset(
        test_iterable, max_samples // 2 if max_samples else None
    )

    test_loader = DataLoader(
        limited_test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )

    return train_loaders, test_loader


# ------------------------------
# Federated tabular loaders
# ------------------------------


def create_federated_tabular_loaders(
    dataset_name: str,
    partitioner: Partitioner,
    batch_size: int,
    num_workers: int,
    task: str,
    preprocessing_strategy,
    max_samples: Optional[int] = None,
) -> Tuple[List[DataLoader], DataLoader]:
    """Create federated loaders for tabular datasets (binary classification, regression)."""

    fds = FederatedDataset(dataset=dataset_name, partitioners={"train": partitioner})
    fds.load_partition(0, "train")  # Trigger download
    train_loaders = []

    for client_id in range(partitioner.num_partitions):
        partition = fds.load_partition(client_id, "train")
        features, labels = extract_features_labels(partition, task)
        features, labels = preprocessing_strategy.preprocess(features, labels)

        if max_samples:
            features, labels = features[:max_samples], labels[:max_samples]

        loader = create_dataloader(features, labels, batch_size, num_workers)
        train_loaders.append(loader)

    # Test set (not partitioned)
    full_dataset = datasets.load_dataset(dataset_name)
    if "test" in full_dataset:
        test_dataset = full_dataset["test"]
    else:
        train_test = full_dataset["train"].train_test_split(test_size=0.2)
        test_dataset = train_test["test"]

    test_features, test_labels = extract_features_labels(test_dataset, task)
    test_features, test_labels = preprocessing_strategy.preprocess(
        test_features, test_labels
    )

    if max_samples:
        test_features, test_labels = (
            test_features[: max_samples // 2],
            test_labels[: max_samples // 2],
        )

    test_loader = create_dataloader(
        test_features, test_labels, batch_size, num_workers, shuffle=False
    )

    return train_loaders, test_loader


# ------------------------------
# Dataset factory
# ------------------------------


class DatasetFactory:
    """Factory class for creating dataset configs."""

    DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
        "image_classification": {
            "name": "uoft-cs/cifar10",  # <-- updated
            "strategy": None,  # preprocessing handled in collate_fn
            "max_samples": 1000,
        },
        "binary_classification": {
            "name": "mnemoraorg/wisconsin-breast-cancer-diagnostic",
            "strategy": BinaryClassificationPreprocessing(),
            "max_samples": None,
        },
        "regression": {
            "name": "SIH/palmer-penguins",
            "strategy": RegressionPreprocessing(),
            "max_samples": None,
        },
    }

    @classmethod
    def create_dataset(cls, task: str) -> Tuple[str, Any, Optional[int]]:
        if task not in cls.DATASET_CONFIGS:
            raise ValueError(
                f"Unsupported task: {task}. Choose from {list(cls.DATASET_CONFIGS.keys())}"
            )
        config = cls.DATASET_CONFIGS[task]
        return config["name"], config["strategy"], config["max_samples"]


# ------------------------------
# Main entrypoint
# ------------------------------


def datamaker(
    task: str,
    partitioner: Partitioner,
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int] = None,
) -> Tuple[Iterable[DataLoader], DataLoader]:
    """Create train and test DataLoaders for the specified task."""

    dataset_name, preprocessing_strategy, default_max = DatasetFactory.create_dataset(
        task
    )
    max_samples = max_samples or default_max

    if task == "image_classification":
        return create_federated_cifar10_loaders(
            dataset_name=dataset_name,
            partitioner=partitioner,
            batch_size=batch_size,
            num_workers=num_workers,
            max_samples=max_samples,
        )
    else:  # binary_classification / regression
        return create_federated_tabular_loaders(
            dataset_name=dataset_name,
            partitioner=partitioner,
            batch_size=batch_size,
            num_workers=num_workers,
            task=task,
            preprocessing_strategy=preprocessing_strategy,
            max_samples=max_samples,
        )
