import logging
import os
from os.path import exists

def set_logger(log_path: str):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    while logger.handlers:
        logger.handlers.pop()

    # Logging to file
    if exists(log_path):
        os.remove(log_path)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(funcName)s:%(lineno)d | %(message)s"
        )
    )
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(funcName)s:%(lineno)d | %(message)s"
        )
    )
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)


if __name__ == "__main__":
    log_path = "./logger_test.log"
    set_logger(log_path)
    logging.info("Logging one line")
    logging.info("Logging two lines")
    logging.info("Logging three lines")
    logging.warning("Warning one")
    logging.warning("Warning two")
    logging.error("Error one")
    logging.error("Error two")
