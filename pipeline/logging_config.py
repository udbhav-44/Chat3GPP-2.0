import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    date_format = os.getenv("LOG_DATEFMT", "%Y-%m-%dT%H:%M:%S%z")

    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(log_format, datefmt=date_format)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    log_file = os.getenv("LOG_FILE")
    if log_file:
        max_bytes = int(os.getenv("LOG_MAX_BYTES", "10485760"))
        backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    noisy_level_name = os.getenv("LOG_NOISY_LEVEL", "WARNING").upper()
    noisy_level = getattr(logging, noisy_level_name, logging.WARNING)
    for noisy_logger in ("httpx", "openai", "websockets", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(noisy_level)

    logging.captureWarnings(True)
