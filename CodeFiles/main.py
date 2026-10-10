import logging
import sys
from logging.handlers import RotatingFileHandler
import raw_sampling


def setup_logging(log_file="milestone2.log", level=logging.INFO):
    """Configures the root logger for the entire application."""

    # 1. Define a common format for log messages
    log_format = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 2. Setup Console Handler (Outputs to terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)

    # 3. Setup Rotating File Handler (Keeps logs up to 5MB, rotates up to 3 files)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(log_format)

    # 4. Configure the Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logs in some environments
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

setup_logging(log_file="milestone2.log", level=logging.INFO)

logger = logging.getLogger()


if __name__ == "__main__":
    logger.info("Application has started successfully.")
    raw_sample = raw_sampling.sampling()
    output = raw_sampling.process_df(raw_sample)
    output.to_csv("sampleddata.csv", index=False)
    logger.info("Application has finished successfully.")
