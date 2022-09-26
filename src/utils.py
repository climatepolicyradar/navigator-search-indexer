"""Utility functions."""

from typing import List
import logging
import sys
import time


def get_timestamp() -> str:
    """Return a string representing the current date and time."""
    return time.strftime("%Y%m%d-%H%M")


def paginate_list(_list: list, page_size: int) -> List[list]:
    """Split a list into sublists of size page_size. If the list length isn't a multiple of page_size, the last sublist will be smaller.

    Args:
        _list (list): input list
        page_size (int): size of sublists to return.

    Returns:
        List[list]: A list of sublists.
    """
    return [_list[i : i + page_size] for i in range(0, len(_list), page_size)]


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_string)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
