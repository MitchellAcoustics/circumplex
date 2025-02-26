from typing import List, Tuple, Union, Optional, Any

import numpy as np
import pandas as pd

# Import Instrument type for type annotation only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from circumplex.instrument import Instrument

# Common constants
OCTANTS = (0, 45, 90, 135, 180, 225, 270, 315)


def cosine_form(theta: np.ndarray, ampl: float, disp: float, elev: float) -> np.ndarray:
    """
    Cosine function with amplitude, displacement and elevation parameters.
    
    This is the mathematical model used in the Structural Summary Method.
    
    Parameters
    ----------
    theta : np.ndarray
        Angular positions in radians.
    ampl : float
        Amplitude of the cosine curve.
    disp : float
        Angular displacement in radians.
    elev : float
        Elevation (mean level) of the cosine curve.
        
    Returns
    -------
    np.ndarray
        Predicted values at each theta position.
    """
    return elev + ampl * np.cos(theta - disp)


def angle_median(angles: np.ndarray) -> float:
    """
    Calculate the median of circular data.

    Args:
        angles (np.ndarray): Array of angles in radians.

    Returns:
        float: Median angle in radians.
    """
    return np.arctan2(np.median(np.sin(angles)), np.median(np.cos(angles)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the R² coefficient of determination for a set of predictions.
    
    Measures how well the predictions match the observed data.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
        
    Returns
    -------
    float
        R² value between 0 and 1, where 1 indicates perfect prediction.
    """
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - (ss_res / ss_tot)


def sort_angles(
        angles: np.ndarray, scores: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Sort angles and corresponding scores in ascending order."""
    sorted_indices = np.argsort(angles)
    return np.array(angles)[sorted_indices], np.array(scores)[sorted_indices]


def standardize(
        data: pd.DataFrame,
        scales: Union[List[str], pd.Index],
        angles: List[float],
        instrument: 'Instrument',
        sample: int = 1,
        prefix: str = "",
        suffix: str = "_z",
        ) -> pd.DataFrame:
    """
    Standardize circumplex scales using normative data.

    Args:
        data (pd.DataFrame): A DataFrame containing at least circumplex scales.
        scales (Union[List[str], pd.Index]): The column names for the variables in data that contain circumplex scales to be standardized.
        angles (List[float]): A numeric list containing the angular displacement of each circumplex scale (in degrees).
        instrument (Instrument): An instrument object containing normative data.
        sample (int): An integer corresponding to the normative sample to use in standardizing the scale scores (default = 1).
        prefix (str): A string to include at the beginning of the newly calculated scale variables' names (default = "").
        suffix (str): A string to include at the end of the newly calculated scale variables' names (default = "_z").

    Returns:
        pd.DataFrame: A DataFrame that matches data except that new variables are appended that contain standardized versions of scales.
    """
    assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"
    assert all(
            scale in data.columns for scale in scales
            ), "All scales must be present in data"
    assert len(scales) == len(angles), "scales and angles must have the same length"
    assert isinstance(sample, int), "sample must be an integer"
    assert isinstance(prefix, str), "prefix must be a string"
    assert isinstance(suffix, str), "suffix must be a string"

    norms = instrument.norms
    key = norms.table.query("sample == @sample")
    assert len(key) == len(scales)

    for scale, angle in zip(scales, angles):
        new_var = f"{prefix}{scale}{suffix}"
        m = key.query("scale == @scale")["m"].values[0]
        sd = key.query("scale == @scale")["sd"].values[0]
        data[new_var] = (data[scale] - m) / sd

    return data
