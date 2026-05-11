"""Reproducible random seeding for Python, NumPy, and PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed all RNGs we care about.

    When ``deterministic=True`` we also flip the PyTorch deterministic-algos
    switch. This makes training slower but bit-for-bit reproducible. Default
    is off because production training values throughput.
    """

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch is optional at import time so utility scripts don't pay for it.
        pass
