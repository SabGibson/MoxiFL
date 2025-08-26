class DeviceLogger:
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.history: dict[str, list[float]] = {}

    def log(self, metric: str, value: float, step: int | None = None):
        if metric not in self.history:
            self.history[metric] = []
        self.history[metric].append((step, value))

    def get_history(self):
        return self.history

    def clear_history(self):
        self.history = {}
