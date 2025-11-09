"""Helper utilities for circumplex visualization."""

import numpy as np


def ggrad(degrees: float | np.ndarray) -> float | np.ndarray:
    """Convert degrees to ggplot2-style radians.

    ggplot2 uses a coordinate system where:
    - 90° is at the top (North)
    - Angles proceed clockwise

    This function converts standard circumplex degrees to this system.

    Parameters
    ----------
    degrees
        Angle(s) in degrees (0-360).

    Returns
    -------
    Angle(s) in radians with ggplot2 orientation.

    Examples
    --------
    >>> ggrad(90)   # Top (North)
    0.0
    >>> ggrad(0)    # Right (East)
    1.5707963267948966
    >>> ggrad(180)  # Left (West)
    -1.5707963267948966

    """
    return (degrees - 90) * (-np.pi / 180)


def pretty_max(values: np.ndarray) -> float:
    """Find a nice maximum value for amplitude scaling.

    Selects from a predefined set of "pretty" values that are slightly
    larger than the maximum data value, providing appropriate buffer space.

    Parameters
    ----------
    values
        Array of amplitude values (typically upper CI bounds).

    Returns
    -------
    A nice round number suitable for the amplitude scale maximum.

    Examples
    --------
    >>> pretty_max(np.array([0.42, 0.38, 0.45]))
    0.75
    >>> pretty_max(np.array([1.2, 1.5, 1.3]))
    2.5

    """
    amax = np.nanmax(values)

    options = np.array(
        [
            -5.00,
            -4.00,
            -3.00,
            -2.50,
            -2.00,
            -1.50,
            -1.25,
            -1.00,
            -0.75,
            -0.50,
            -0.25,
            -0.20,
            -0.15,
            -0.10,
            -0.05,
            0.00,
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.50,
            0.75,
            1.00,
            1.25,
            1.50,
            2.00,
            2.50,
            3.00,
            4.00,
            5.00,
        ]
    )

    # If negative, use smaller buffer; if positive, use larger buffer
    scalar = 0.5 if amax < 0 else 1.5

    # Find the first option larger than amax * scalar
    idx = np.searchsorted(options, amax * scalar)

    return options[min(idx, len(options) - 1)]
