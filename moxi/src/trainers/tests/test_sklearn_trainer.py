from sklearn.linear_model import LinearRegression
from moxi.src.trainers.sklearn_trainer import ScikitLearnTrainer
from sklearn.utils.validation import check_is_fitted
import pytest


def test_create_sklearn_trainer(sample_linear_regression_setup):
    """Test creating a Scikit-learn trainer."""
    # Given
    device_id = "device_1"
    model = LinearRegression()
    data = sample_linear_regression_setup[0]
    config = sample_linear_regression_setup[1]
    # When
    trainer = ScikitLearnTrainer(device_id, model, data, config)

    # Then
    assert isinstance(trainer, ScikitLearnTrainer)
    assert trainer.device_id == device_id


def test_train_model_with_trainer(sample_linear_regression_setup):
    """Test training a model with Scikit-learn trainer."""
    # Given
    device_id = "device_1"
    model = LinearRegression()
    data = sample_linear_regression_setup[0]
    config = sample_linear_regression_setup[1]
    with pytest.raises(Exception):
        check_is_fitted(model)

    # When
    trainer = ScikitLearnTrainer(device_id, model, data, config)
    trainer.train()
    # Then
    assert isinstance(trainer, ScikitLearnTrainer)
    assert trainer.state["train"] is True
    check_is_fitted(trainer.model)


def test_evaluate_model_with_trainer(sample_sklearn_trainer_trained):
    """Test evaluating a model with Scikit-learn trainer."""
    # Given
    trainer = sample_sklearn_trainer_trained

    # When
    trainer.evaluate()
    # Then
    assert isinstance(trainer, ScikitLearnTrainer)
    assert trainer.state["evaluate"] is True
    assert "mse_val" in trainer.logger.get_history()
    assert len(trainer.logger.get_history()["mse_val"]) > 0


def test_round_update_property(sample_sklearn_trainer_evaluated):
    """Test round update and property of a model with Scikit-learn trainer."""
    # Given
    trainer = sample_sklearn_trainer_evaluated
    current_round = trainer.round

    # When
    trainer.update_round()
    # Then
    assert isinstance(trainer, ScikitLearnTrainer)
    assert trainer.round == current_round + 1


def test_round_update_method_protected(sample_sklearn_trainer_trained):
    """Test round update method is protected of a model with Scikit-learn trainer."""
    # Given
    trainer = sample_sklearn_trainer_trained
    current_round = trainer.round

    # When
    trainer.update_round()
    # Then
    assert isinstance(trainer, ScikitLearnTrainer)
    assert trainer.round == current_round


def test_logging_functionality(sample_sklearn_trainer_evaluated, mock_mlflow):
    """Test logging functionality of a model with Scikit-learn trainer."""
    # Given
    trainer = sample_sklearn_trainer_evaluated
    initial_round = trainer.round
    initial_log_length = len(trainer.logger.get_history().get("mse_train", []))

    # When
    trainer.log()
    # Then
    mock_mlflow["start_run"].assert_called_once()
