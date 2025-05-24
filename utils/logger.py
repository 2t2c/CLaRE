import logging
import sys

LOGGER = logging.getLogger("dataset_manifest")
LOGGER.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)

if not LOGGER.hasHandlers():
    LOGGER.addHandler(console_handler)
