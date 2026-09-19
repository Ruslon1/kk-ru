from __future__ import annotations

import logging
import random
import time
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name: str = "kk_ru", level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    return logging.getLogger(name)


def human(n: float) -> str:
    units = ["", "K", "M", "B", "T"]
    for unit in units:
        if abs(n) < 1000 or unit == units[-1]:
            return f"{n:.2f}{unit}" if unit else f"{n:.0f}"
        n /= 1000
    return f"{n:.2f}T"


@contextmanager
def timer(label: str, logger: logging.Logger | None = None):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    message = f"{label}: {elapsed:.2f}s"
    if logger is not None:
        logger.info(message)
    else:
        print(message)
