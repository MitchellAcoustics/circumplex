"""Utility functions for data tidying and instrument scoring."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from circumplex.instruments import get_instrument
from circumplex.instruments.models import Instrument


def ipsatize(
    data: pd.DataFrame,
    items: Iterable[str | int],
    prefix: str = "",
    suffix: str = "_i",
    *,
    na_rm: bool = True,
    append: bool = True,
) -> pd.DataFrame:
    """Ipsatize item-level data by centering within individuals.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing item-level data.
    items : tuple[str]
        Tuple of column names corresponding to item-level data to ipsatize.
    prefix : str, optional
        Prefix to add to ipsatized column names, by default "".
    suffix : str, optional
        Suffix to add to ipsatized column names, by default "".
    na_rm : bool, optional
        Whether to remove NAs when computing individual means, by default True.
    append : bool, optional
        Whether to append ipsatized columns to the original DataFrame,
        or return only the ipsatized columns, by default True.

    Returns
    -------
    DataFrame with ipsatized item-level data.

    Raises
    ------
    TypeError
        If `data` is not a DataFrame or `items` is not a sequence.
    ValueError
        If any item in `items` is not a column in `data`.
    """
    if not isinstance(data, pd.DataFrame):
        msg = "Input 'data' must be a pandas DataFrame."
        raise TypeError(msg)
    if isinstance(items, str):
        msg = "Input 'items' must be a sequence of column names."
        raise TypeError(msg)
    if not isinstance(items, Iterable):
        msg = "Input 'items' must be a sequence of column names."
        raise TypeError(msg)

    if all(isinstance(item, str) for item in items):
        if all(item in data.columns for item in items):
            item_data = data.loc[:, list(items)].copy()
        else:
            msg = "All items in 'items' must be valid column names in 'data'."
            raise ValueError(msg)
    elif all(isinstance(item, (int, np.integer, float, np.floating)) for item in items):
        numeric_items = [int(item) for item in items]
        if all(0 <= idx < data.shape[1] for idx in numeric_items):
            item_data = data.iloc[:, numeric_items].copy()
        else:
            msg = "All items in 'items' must be valid indices in 'data'."
            raise ValueError(msg)
    else:
        msg = "All items in 'items' must be either strings or integers."
        raise TypeError(msg)

    rmean = item_data.mean(axis=1, skipna=na_rm)
    scores = item_data.subtract(rmean, axis=0)
    scores.columns = [f"{prefix}{item}{suffix}" for item in items]

    if append:
        return pd.concat([data, scores], axis=1)
    return scores


def score(
    data: pd.DataFrame,
    items: Iterable[str | int],
    instrument: Instrument | str,
    prefix: str = "",
    suffix: str = "",
    *,
    na_rm: bool = True,
    append: bool = True,
) -> pd.DataFrame:
    """Score item-level data using a circumplex instrument.

    Parameters
    ----------
    data
        DataFrame containing at least circumplex scales.
    items
        The variable names or column numbers for the variables in `data`
        that contain all the circumplex items from a single circumplex measure,
        in ascending order from item 1 to item N.
    instrument
        An instrument object from the package. To see the available
    prefix : str, optional
        Prefix to add to scored column names, by default "".
    suffix : str, optional
        Suffix to add to scored column names, by default "".
    na_rm : bool, optional
        Whether to remove NAs when computing individual means, by default True.
    append : bool, optional
        Whether to append scored columns to the original DataFrame,
        or return only the scored columns, by default True.

    Returns
    -------
    DataFrame with scored scale-level data.

    Raises
    ------
    TypeError
        If `data` is not a DataFrame or `items` is not a sequence.
    ValueError
        If any item in `items` is not a column in `data`.
    """
    # Validate inputs
    # -- validate data and items first (before trying to get instrument)
    if not isinstance(data, pd.DataFrame):
        msg = "Input 'data' must be a pandas DataFrame."
        raise TypeError(msg)
    if isinstance(items, str):
        msg = "Input 'items' must be a sequence of column names."
        raise TypeError(msg)
    if not isinstance(items, Iterable):
        msg = "Input 'items' must be a sequence of column names or indices."
        raise TypeError(msg)

    # -- get instrument if a string is provided
    if isinstance(instrument, str):
        instrument = get_instrument(instrument)
    if not isinstance(instrument, Instrument):
        msg = "Input 'instrument' must be an Instrument instance."
        raise TypeError(msg)

    return instrument.score(
        data,
        items,
        prefix=prefix,
        suffix=suffix,
        na_rm=na_rm,
        append=append,
    )


def norm_standardize(
    data: pd.DataFrame,
    instrument: Instrument | str,
    sample_id: int,
    scales: Iterable[str | int] | None = None,
    prefix: str = "",
    suffix: str = "_z",
    *,
    append: bool = True,
) -> pd.DataFrame:
    """Standardize scale-level data using normative sample statistics.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing scale-level data.
    scales : tuple[str | int]
        Tuple of column names or indices corresponding to scale-level data
        to standardize.
    instrument : Instrument | str
        An instrument object from the package. To see the available
        instruments, use `show_instruments()`.
    sample_id : int | str
        The ID of the normative sample to use for standardization.
    prefix : str, optional
        Prefix to add to standardized column names, by default "".
    suffix : str, optional
        Suffix to add to standardized column names, by default "_z".
    append : bool, optional
        Whether to append standardized columns to the original DataFrame,
        or return only the standardized columns, by default True.

    Returns
    -------
    DataFrame with standardized scale-level data.

    Raises
    ------
    TypeError
        If `data` is not a DataFrame or `scales` is not a sequence.
    ValueError
        If any scale in `scales` is not a column in `data`.
    """
    # Validate inputs
    # -- validate data and scales first (before trying to get instrument)
    if not isinstance(data, pd.DataFrame):
        msg = "Input 'data' must be a pandas DataFrame."
        raise TypeError(msg)
    if isinstance(scales, str):
        msg = "Input 'scales' must be a sequence of column names."
        raise TypeError(msg)
    if not isinstance(scales, Iterable) and scales is not None:
        msg = "Input 'scales' must be a sequence of column names or indices."
        raise TypeError(msg)

    # -- get instrument if a string is provided
    if isinstance(instrument, str):
        instrument = get_instrument(instrument)
    if not isinstance(instrument, Instrument):
        msg = "Input 'instrument' must be an Instrument instance."
        raise TypeError(msg)

    return instrument.norm_standardize(
        data,
        int(sample_id),
        scales=scales,
        prefix=prefix,
        suffix=suffix,
        append=append,
    )
