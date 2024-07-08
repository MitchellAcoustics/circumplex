import numpy as np


def cosine_form(theta, ampl, disp, elev):
    """Cosine function with amplitude, dispersion and elevation parameters."""
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


def r2_score(y_true: np.array, y_pred: np.array):
    """Calculate the R2 score for a set of predictions."""
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - (ss_res / ss_tot)


OCTANTS = (0, 45, 90, 135, 180, 225, 270, 315)
