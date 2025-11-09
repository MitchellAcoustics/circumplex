"""Angle utilities and predefined angle sets.

This module provides tools for working with circular angles, including
conversion between degrees and radians, and standard angle sets for
circumplex models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nptyping import Float, NDArray, Shape


OCTANTS: NDArray[Shape["8"], Float] = np.array(
    [90, 135, 180, 225, 270, 315, 360, 45], dtype=float
)
"""Standard octant angles in degrees.

Returns the eight standard positions on a circumplex circle,
spaced 45 degrees apart, starting from 90 degrees (North).
"""

QUADRANTS: NDArray[Shape["4"], Float] = np.array([90, 180, 270, 360], dtype=float)
"""Standard quadrant angles in degrees.

Returns the four standard quadrant positions on a circumplex circle,
spaced 90 degrees apart.
"""

POLES: NDArray[Shape["2"], Float] = np.array([90, 270], dtype=float)
"""Standard pole angles in degrees.

Returns the two primary axis positions (vertical poles)
on a circumplex circle.
"""


class Degree(float):
    """Angular measurement in degrees.

    A float subclass representing an angle in degrees, with
    conversion methods to radians.

    Examples
    --------
    >>> angle = Degree(90)
    >>> angle.to_radians()
    Radian(1.5707963267948966)

    """

    def to_radians(self) -> "Radian":
        """Convert to radians.

        Returns
        -------
        Angle in radians

        """
        return Radian(np.radians(self))

    def __repr__(self) -> str:
        """Return a compact string representation, e.g., '90°'."""
        return f"{float(self):.0f}°"


class Radian(float):
    """Angular measurement in radians.

    A float subclass representing an angle in radians, with
    conversion methods to degrees.

    Examples
    --------
    >>> angle = Radian(np.pi/2)
    >>> angle.to_degrees()
    Degree(90.0)

    """

    def to_degrees(self) -> Degree:
        """Convert to degrees.

        Returns
        -------
        Angle in degrees

        """
        return Degree(np.degrees(self))

    def __repr__(self) -> str:
        """Return a compact string representation, e.g., '1.571 rad'."""
        return f"{float(self):.3f} rad"


def degrees_to_radians(degrees: float | np.ndarray) -> float | np.ndarray:
    """Convert degrees to radians.

    Parameters
    ----------
    degrees
        Angle(s) in degrees

    Returns
    -------
    Angle(s) in radians

    Examples
    --------
    >>> degrees_to_radians(180)
    3.141592653589793
    >>> degrees_to_radians(OCTANTS)
    array([1.57..., 2.35..., 3.14..., ...])

    """
    return np.radians(degrees)


def radians_to_degrees(radians: float | np.ndarray) -> float | np.ndarray:
    """Convert radians to degrees.

    Parameters
    ----------
    radians
        Angle(s) in radians

    Returns
    -------
    Angle(s) in degrees

    Examples
    --------
    >>> radians_to_degrees(np.pi)
    180.0

    """
    return np.degrees(radians)
