from abc import ABC, abstractmethod


class AbstractMoxiTrainer(ABC):
    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abstractmethod
    def log(self, *args, **kwargs):
        pass
