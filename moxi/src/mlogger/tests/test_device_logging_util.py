from moxi.src.mlogger.device_logger import DeviceLogger


def test_create_device_logger():
    """Test creating a device logger."""
    # Given
    device_id = "device_123"
    # When
    logger = DeviceLogger(device_id)
    # Then
    assert isinstance(logger, DeviceLogger)
    assert logger.device_id == device_id


def test_log_new_metric():
    """Test logging a new metric."""
    # Given
    logger = DeviceLogger("device_123")
    metric = "accuracy"
    value = 0.95
    step = 1
    # When
    logger.log(metric, value, step)
    # Then
    history = logger.get_history()
    assert metric in history
    assert history[metric] == [(step, value)]


def test_log_existing_metric():
    """Test logging an existing metric."""
    # Given
    logger = DeviceLogger("device_123")
    metric = "loss"
    values = [0.5, 0.4, 0.3]
    steps = [1, 2, 3]
    for step, value in zip(steps, values):
        logger.log(metric, value, step)
    # When
    new_value = 0.2
    new_step = 4
    logger.log(metric, new_value, new_step)
    # Then
    history = logger.get_history()
    assert metric in history
    assert history[metric] == list(zip(steps + [new_step], values + [new_value]))


def test_clear_history():
    """Test clearing the history."""
    # Given
    logger = DeviceLogger("device_123")
    metric = "accuracy"
    value = 0.95
    step = 1
    logger.log(metric, value, step)
    # When
    logger.clear_history()
    # Then
    history = logger.get_history()
    assert history == {}
