from .async_worker import AsyncMoxiPytorchWorker

WORKER_MAP = {"async": {"pytorch": AsyncMoxiPytorchWorker}, "sync": {}}
