import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm


def group_values(values: np.ndarray, groups: np.ndarray, group_name: str) -> np.ndarray:
    """
    Filter values array by group indices and clean it from NaNs.

    :param values: Values to filter.
    :param groups: Timeseries groups.
    :param group_name: Group name to filter by.
    :return: Filtered and cleaned timeseries.
    """
    return np.array([v[~np.isnan(v)] for v in values[groups == group_name]])