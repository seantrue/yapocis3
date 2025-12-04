#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import logging
import logging.handlers


def get_logger(
    name: str = "yapocis",
    format: str = "%(asctime)s  %(levelname)s %(module)s %(funcName)s %(lineno)s :: %(message)s",
    level=logging.INFO,
):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(format, datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    handler = logging.handlers.RotatingFileHandler(
        f"{name}.log", mode="a", maxBytes=10_000_000, backupCount=5
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(handler)
    return logger
