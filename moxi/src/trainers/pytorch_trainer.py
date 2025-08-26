from .abstractmoxitrainer import AbstractMoxiTrainer
from .interfaces import MoxiTrainerConfig, MoxiTrainerFrameworkType, MoxiTorchDataset
from moxi.src.mlogger import DeviceLogger
from moxi.src.util.pytroch_util import simple_train_model, simple_eval_model
import mlflow
import torch


class PytorchTrainer(AbstractMoxiTrainer):
    def __init__(
        self,
        device_id: str,
        model: any,
        data: MoxiTorchDataset,
        config: MoxiTrainerConfig,
    ) -> None:
        assert (
            config["framework"] == MoxiTrainerFrameworkType.PYTORCH
        ), "Invalid framework type for PytorchTrainer"

        self.device_id = device_id
        self.model = model
        self.data = data
        self.config = config
        self.logger: DeviceLogger = DeviceLogger(device_id)
        self._round: int = 0
        self.state: dict = {"train": False, "evaluate": False}

    def train(self) -> None:
        """Train the model using the training loop."""
        simple_train_model(
            model=self.model,
            train_dataloader=self.data["train_data"],
            criterion=self.config["criterion"][0],
            optimizer_cls=self.config["optimizer"],
            epochs=self.config.get("epochs", 5),
            device=self.config.get("device", "cpu"),
            val_dataloader=self.data.get("val_dataloader", None),
            learning_rate=self.config.get("learning_rate", 0.01),
            logger=self.logger,
        )
        self.state["train"] = True

    def evaluate(self) -> float:
        """Run evaluation loop on validation/test data and return avg loss."""
        val_loader = self.data.get("val_data")
        if val_loader is None:
            raise ValueError("No validation dataloader provided for evaluation.")

        avg_val_loss = simple_eval_model(
            model=self.model,
            dataloader=val_loader,
            criterion=self.config["criterion"][0],
            device=self.config.get("device", "cpu"),
            logger=self.logger,
            step=self.round,
        )

        self.state["evaluate"] = True
        return avg_val_loss

    @property
    def round(self) -> int:
        return self._round

    def update_round(self) -> None:
        """Increment round counter when both train and evaluate have run."""
        if self.state["train"] and self.state["evaluate"]:
            self._round += 1
            self.state = {"train": False, "evaluate": False}

    def log(self) -> None:
        """Log metrics & params to MLflow for this round."""
        with mlflow.start_run(
            run_name=f"round_{self.round}",
            experiment_id=self.device_id,
        ):
            # log params from config
            mlflow.log_param("epochs", self.config.get("epochs", 5))
            mlflow.log_param("optimizer", self.config["optimizer"].__name__)
            mlflow.log_param("criterion", type(self.config["criterion"]).__name__)
            mlflow.log_param("learning_rate", self.config.get("learning_rate", 0.01))

            # log metrics from DeviceLogger
            for metric, values in self.logger.get_history().items():
                for step, value in values:
                    mlflow.log_metric(
                        metric, value, step if step is not None else self.round
                    )

            # mlflow.pytorch.log_model(self.model, artifact_path="model")
