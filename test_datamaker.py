import torch
from flwr_datasets.partitioner import IidPartitioner
from moxi import datamaker


def inspect_loaders(task: str):
    print(f"\n=== Testing {task} ===")
    partitioner = IidPartitioner(num_partitions=2)
    train_loaders, test_loader = datamaker(
        task=task,
        partitioner=partitioner,
        batch_size=4,
        num_workers=0,
        max_samples=20,  # keep tiny for test
    )

    # Inspect first batch from first client
    x, y = next(iter(train_loaders[0]))
    print(f"Train batch X shape: {x.shape}, Y shape: {y.shape}")
    x_test, y_test = next(iter(test_loader))
    print(f"Test batch X shape: {x_test.shape}, Y shape: {y_test.shape}")


if __name__ == "__main__":
    for task in ["binary_classification", "regression", "image_classification"]:
        inspect_loaders(task)
