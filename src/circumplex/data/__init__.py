"""Example datasets and data loading utilities.

This module provides access to the built-in example datasets that
are used for testing and demonstration purposes.
"""

from pathlib import Path
from typing import Literal

import pandas as pd

# Type for available datasets
DatasetName = Literal["jz2017", "aw2009"]


def load_dataset(name: DatasetName) -> pd.DataFrame:
    """Load a built-in example dataset.

    Parameters
    ----------
    name : str
        Dataset name. Available datasets:
        - 'jz2017': Interpersonal problems and personality disorder data
        - 'aw2009': Standardized octant scores

    Returns
    -------
    pd.DataFrame
        The requested dataset

    Raises
    ------
    ValueError
        If dataset name is not recognized

    Examples
    --------
    >>> from circumplex.data import load_dataset
    >>> jz2017 = load_dataset('jz2017')
    >>> print(jz2017.shape)
    (1166, 19)
    >>> print(jz2017.columns[:9].tolist())
    ['Gender', 'PA', 'BC', 'DE', 'FG', 'HI', 'JK', 'LM', 'NO']

    Notes
    -----
    The jz2017 dataset contains data from Zimmermann & Wright (2017):
    - 1166 observations
    - 8 IIP-SC octant scales (PA through NO)
    - 10 PDQ-4+ personality disorder scales
    - Gender grouping variable

    The aw2009 dataset contains data from Wright et al. (2009):
    - 5 observations
    - 8 standardized circumplex scales

    """
    valid_datasets = {"jz2017", "aw2009"}

    if name not in valid_datasets:
        msg = (
            f"Unknown dataset '{name}'. Valid options are: {', '.join(valid_datasets)}"
        )
        raise ValueError(msg)

    # Path to data files
    data_dir = Path(__file__).parent
    filepath = data_dir / f"{name}.csv"

    if not filepath.exists():
        msg = (
            f"Dataset file not found: {filepath}. "
            f"This may indicate a corrupted package installation."
        )
        raise FileNotFoundError(msg)

    return pd.read_csv(filepath)


def list_datasets() -> list[str]:
    """List all available built-in datasets.

    Returns
    -------
    list of str
        Names of available datasets

    Examples
    --------
    >>> from circumplex.data import list_datasets
    >>> datasets = list_datasets()
    >>> print(datasets)
    ['aw2009', 'jz2017']

    """
    return ["aw2009", "jz2017"]
