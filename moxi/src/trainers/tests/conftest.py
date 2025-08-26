import pytest
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from moxi.src.trainers.util import convert_str_to_framework
from unittest.mock import patch, MagicMock
from moxi.src.util import create_dataloader, LinearModel
import torch.nn.functional as F
import torch.optim as optim

from sklearn.linear_model import LinearRegression
from moxi.src.trainers import ScikitLearnTrainer, PytorchTrainer

RANDOM_SEED = 44


@pytest.fixture
def sample_regression_data():
    """Fixture to provide sample regression data for tests."""
    X, y = make_blobs(n_samples=100, centers=3, n_features=5, random_state=RANDOM_SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED
    )
    return X_train, X_val, y_train, y_val


@pytest.fixture
def sample_linear_regression_setup():
    """Fixture to provide sample data for tests."""
    X, y = make_blobs(n_samples=100, centers=3, n_features=5, random_state=RANDOM_SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED
    )
    dataset = {
        "train_data": X_train,
        "train_labels": y_train,
        "val_data": X_val,
        "val_labels": y_val,
    }
    config = {
        "framework": convert_str_to_framework("sklearn"),
        "seed": RANDOM_SEED,
        "metrics": {"mse": mean_squared_error},
    }

    return dataset, config


@pytest.fixture
def sample_sklearn_trainer_trained(sample_linear_regression_setup):

    device_id = "device_1"
    model = LinearRegression()
    data = sample_linear_regression_setup[0]
    config = sample_linear_regression_setup[1]

    trainer = ScikitLearnTrainer(device_id, model, data, config)
    trainer.train()
    return trainer


@pytest.fixture
def sample_sklearn_trainer_evaluated(sample_sklearn_trainer_trained):
    trainer = sample_sklearn_trainer_trained
    trainer.evaluate()
    return trainer


@pytest.fixture
def mock_mlflow():
    """Fixture to mock MLflow for all tests that need it."""
    with patch("mlflow.start_run") as mock_start_run, patch(
        "mlflow.log_metric"
    ) as mock_log_metric, patch("mlflow.log_params") as mock_log_params:

        mock_run = MagicMock()
        mock_start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_start_run.return_value.__exit__ = MagicMock(return_value=False)

        yield {
            "start_run": mock_start_run,
            "log_metric": mock_log_metric,
            "log_params": mock_log_params,
        }


@pytest.fixture
def sample_linear_model_dataset_pytorch(sample_regression_data):
    """Fixture to provide a sample linear model and dataset for PyTorch tests."""

    train_data, val_data, train_labels, val_labels = sample_regression_data
    train_dataloader = create_dataloader(
        train_data, train_labels, batch_size=16, shuffle=True
    )
    val_dataloader = create_dataloader(
        val_data, val_labels, batch_size=16, shuffle=False
    )
    input_dim = train_data.shape[1]
    output_dim = 1
    model = LinearModel(input_dim, output_dim)
    config = {
        "framework": convert_str_to_framework("pytorch"),
        "n_epochs": 5,
        "criterion": [F.mse_loss],
        "optimizer": optim.SGD,
        "learning_rate": 0.01,
    }
    return model, {"train_data": train_dataloader, "val_data": val_dataloader}, config


@pytest.fixture
def sample_linear_model_trained_pytorch(sample_linear_model_dataset_pytorch):
    """Fixture to provide a sample lineartrained model PyTorch tests."""

    model, data, config = sample_linear_model_dataset_pytorch
    device_id = "device_2"
    trainer = PytorchTrainer(device_id, model, data, config)
    trainer.train()
    return trainer
