import logging
import sys

LOGGER = logging.getLogger("fomo")
LOGGER.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)

if not LOGGER.hasHandlers():
    LOGGER.addHandler(console_handler)


def set_logging_level(level: str):
    log_level_mapper = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    log_level = log_level_mapper.get(level)

    if log_level is None:
        choices = list(log_level_mapper.values())
        raise ValueError(f"Invalid logging level! Choices: {choices}")

    LOGGER.setLevel(log_level)
    console_handler.setLevel(log_level)
