"""Contrast and circular difference utilities for SSM analysis.

This module provides functions for computing parameter differences and
circular distance calculations, particularly for contrast analyses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nptyping import Float, Int, NDArray, Shape


def angle_dist(angle1: float, angle2: float) -> float:
    """Calculate circular distance between two angles.

    Computes the signed difference between two angles in radians,
    returning a value in the range [-π, π]. This handles the circular
    nature of angular data correctly.

    Parameters
    ----------
    angle1
        First angle in radians
    angle2
        Second angle in radians

    Returns
    -------
    Signed difference (angle1 - angle2) in radians, range [-π, π]

    Examples
    --------
    >>> angle_dist(0.1, 0.05)  # Small positive difference
    0.05
    >>> angle_dist(0.1, 6.2)  # Wraps around circle
    0.18318...
    >>> angle_dist(np.pi, -np.pi)  # Opposite sides of circle
    0.0

    Notes
    -----
    This function mirrors angle_dist() from the R package (R/utils.R lines 32-34).
    The formula ((x - y + π) mod 2π) - π ensures the result is in [-π, π].

    """
    diff = angle1 - angle2
    # Wrap to [-π, π]
    return ((diff + np.pi) % (2 * np.pi)) - np.pi


def param_diff(
    params1: dict[str, float], params2: dict[str, float]
) -> dict[str, float]:
    """Calculate difference between two sets of SSM parameters.

    Computes element-wise differences between parameter sets, with special
    handling for the displacement parameter (circular difference).

    Parameters
    ----------
    params1
        First parameter set with keys: elevation, x_value, y_value,
        amplitude, displacement, fit
    params2
        Second parameter set with same keys

    Returns
    -------
    Parameter differences (params1 - params2). The displacement difference
    uses circular distance to handle angular wrapping correctly.

    Examples
    --------
    >>> p1 = {'elevation': 1.0, 'x_value': 0.5, 'y_value': 0.3,
    ...       'amplitude': 0.6, 'displacement': 0.1, 'fit': 0.9}
    >>> p2 = {'elevation': 0.8, 'x_value': 0.4, 'y_value': 0.2,
    ...       'amplitude': 0.5, 'displacement': 6.2, 'fit': 0.85}
    >>> diff = param_diff(p1, p2)
    >>> diff['elevation']
    0.2
    >>> diff['displacement']  # Uses circular distance
    0.18318...

    Notes
    -----
    This function mirrors param_diff() from the R package (R/utils.R lines 13-19).
    For displacement, it uses angle_dist() instead of simple subtraction.

    """
    return {
        "elevation": params1["elevation"] - params2["elevation"],
        "x_value": params1["x_value"] - params2["x_value"],
        "y_value": params1["y_value"] - params2["y_value"],
        "amplitude": params1["amplitude"] - params2["amplitude"],
        "displacement": angle_dist(params1["displacement"], params2["displacement"]),
        "fit": params1["fit"] - params2["fit"],
    }


def param_diff_array(
    params: NDArray[Shape["12", Float | Int]],
) -> NDArray[Shape["6", Float | Int]]:
    """Calculate parameter differences from a flat parameter array.

    Given a flat array containing parameters for two groups/measures,
    computes the difference between them (first - second) with circular
    handling for displacement.

    Parameters
    ----------
    params
        Flat array of parameters, length 12:
        [e1, x1, y1, a1, d1, f1, e2, x2, y2, a2, d2, f2]

    Returns
    -------
    Array of parameter differences, length 6:
    [e_diff, x_diff, y_diff, a_diff, d_diff, f_diff]

    Examples
    --------
    >>> params = np.array([1.0, 0.5, 0.3, 0.6, 0.1, 0.9,
    ...                    0.8, 0.4, 0.2, 0.5, 6.2, 0.85])
    >>> diff = param_diff_array(params)
    >>> diff[0]  # Elevation difference
    0.2

    Notes
    -----
    This is a convenience wrapper around param_diff() for use with
    flat arrays returned by group_parameters().

    """
    # Extract first set (indices 0-5) - first group
    p1 = {
        "elevation": params[0],
        "x_value": params[1],
        "y_value": params[2],
        "amplitude": params[3],
        "displacement": params[4],
        "fit": params[5],
    }

    # Extract second set (indices 6-11) - second group
    p2 = {
        "elevation": params[6],
        "x_value": params[7],
        "y_value": params[8],
        "amplitude": params[9],
        "displacement": params[10],
        "fit": params[11],
    }

    # Compute difference as second - first (matches R convention)
    diff = param_diff(p2, p1)  # type: ignore[invalid-argument-type]

    return np.array(
        [
            diff["elevation"],
            diff["x_value"],
            diff["y_value"],
            diff["amplitude"],
            diff["displacement"],
            diff["fit"],
        ]
    )
