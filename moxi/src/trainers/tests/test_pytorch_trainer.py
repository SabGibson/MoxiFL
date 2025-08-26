from moxi.src.trainers.pytorch_trainer import PytorchTrainer
from moxi.src.util import is_trained
import copy


def test_create_pytorch_trainer(sample_linear_model_dataset_pytorch):
    """Test creating a Pytorch trainer."""
    # Given
    model, data, config = sample_linear_model_dataset_pytorch
    device_id = "device_2"
    # When
    trainer = PytorchTrainer(device_id, model, data, config)

    # Then
    assert isinstance(trainer, PytorchTrainer)
    assert trainer.device_id == device_id


def test_train_model_with_trainer(sample_linear_model_dataset_pytorch):
    """Test training a model with Pytorch trainer."""
    # Given
    model, data, config = sample_linear_model_dataset_pytorch
    device_id = "device_2"
    trainer = PytorchTrainer(device_id, model, data, config)
    initial_model = copy.deepcopy(trainer.model)

    # When
    trainer.train()

    # Then
    assert isinstance(trainer, PytorchTrainer)
    assert trainer.state["train"] is True
    assert is_trained(trainer.model, initial_model)


def test_evaluate_model_with_trainer(sample_linear_model_trained_pytorch):
    """Test evaluating a model with pytorch trainer."""
    # Given
    trainer = sample_linear_model_trained_pytorch

    # When
    trainer.evaluate()
    # Then
    assert isinstance(trainer, PytorchTrainer)
    assert trainer.state["evaluate"] is True
    assert "eval_loss" in trainer.logger.get_history()
