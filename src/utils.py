import logging
import inspect

from datetime import datetime
from pathlib import Path
from logging import Logger
from typing import Optional


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


def iprint(msg: str, logger: Optional[Logger] = None, log_queue=None):
    """Informative print."""
    # Get whatever calls it
    frame = inspect.getouterframes(inspect.currentframe())[1]
    func_name = frame.function
    if func_name == 'log':
        frame = inspect.getouterframes(inspect.currentframe())[2]
        func_name = frame.function
    caller_locals = frame.frame.f_locals

    # Check if 'self' or 'cls' is in the local variables to determine if it's a method
    if 'self' in caller_locals:
        class_name = caller_locals['self'].__class__.__name__
        func_name = f"{class_name}.{func_name}"
    elif 'cls' in caller_locals:
        class_name = caller_locals['cls'].__name__
        func_name = f"{class_name}.{func_name}"
    message = f"{func_name}: {msg}"
    assert not (logger and log_queue), 'logger and log_queue are mutually exclusive'

    if log_queue:
        log_queue.put(message)
    elif logger:
        logger.debug(message)
    else:
        print(message)
