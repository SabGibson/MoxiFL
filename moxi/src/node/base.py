from abc import ABC


class AbstractMoxiWorker(ABC):
    def __init__(*args, **kwargs):
        pass

    def train_local(self, *args, **kwargs):
        pass

    def send_updates(self, *args, **kwargs):
        pass

    def compute_update(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        pass
