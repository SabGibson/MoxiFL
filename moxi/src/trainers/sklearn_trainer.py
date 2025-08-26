from .abstractmoxitrainer import AbstractMoxiTrainer
from .interfaces import MoxiTrainerConfig, MoxiTrainerFrameworkType, MoxiDataset
from moxi.src.mlogger import DeviceLogger
import mlflow


class ScikitLearnTrainer(AbstractMoxiTrainer):
    def __init__(
        self, device_id: str, model: any, data: MoxiDataset, config: MoxiTrainerConfig
    ) -> None:
        assert (
            config["framework"] == MoxiTrainerFrameworkType.SCIKIT_LEARN
        ), "Invalid framework type for ScikitLearnTrainer"

        self.device_id = device_id
        self.model = model
        self.data = data
        self.config = config
        self.logger: DeviceLogger = DeviceLogger(device_id)
        self._round: int = 0
        self.state: dict = {"train": False, "evaluate": False}

    def train(self) -> None:
        self.model.fit(self.data["train_data"], self.data["train_labels"])
        for metric_name, metric in self.config.get("metrics", {}).items():
            y_pred = self.model.predict(self.data["train_data"])
            score = metric(self.data["train_labels"], y_pred)
            self.logger.log(metric_name + "_train", score)
        self.state["train"] = True

    def evaluate(
        self,
    ) -> float:
        y_pred = self.model.predict(self.data["val_data"])
        for metric_name, metric in self.config.get("metrics", {}).items():
            score = metric(self.data["val_labels"], y_pred)
            self.logger.log(metric_name + "_val", score)
        self.state["evaluate"] = True

    @property
    def round(self) -> int:
        return self._round

    def update_round(self) -> None:
        if self.state["train"] and self.state["evaluate"]:
            self._round += 1

    def log(self) -> None:
        with mlflow.start_run(
            run_id=f"round_{self.round}", experiment_id=self.device_id
        ):
            for k, v in self.logger.get_history().items():
                mlflow.log_metric(k, v)
        self.update_round()
        self.state["train"] = False
        self.state["evaluate"] = False
