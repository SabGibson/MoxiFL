from .abstractmoxitrainer import AbstractMoxiTrainer


class DummyTrainer(AbstractMoxiTrainer):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def train(self, *args, **kwargs):
        pass

    def evaluate(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass
