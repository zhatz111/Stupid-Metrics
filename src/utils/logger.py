import logging
from pathlib import Path

def get_logger(name: str) -> logging.Logger:

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Get absolute path to the logs directory relative to this file
        base_dir = Path(__file__).resolve().parent.parent.parent  # one level up from utils/
        logs_dir = Path(base_dir, "logs")
        logs_dir.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(logs_dir / "trading.log")
        file_handler.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
