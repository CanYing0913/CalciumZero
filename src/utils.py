import logging

from datetime import datetime
from pathlib import Path


def setup_logger(path) -> logging.Logger:
    log_folder = Path(path).joinpath("log")
    log_folder.mkdir(exist_ok=True)
    log_path = log_folder.joinpath(Path('log_' + datetime.now().strftime("%y%m%d_%H%M%S") + '.txt'))
    logger = logging.getLogger('GUI')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.debug('Logging setup finished.')

    return logger
