"""Angle utilities and predefined angle sets.

This module provides tools for working with circular angles, including
conversion between degrees and radians, and standard angle sets for
circumplex models.
"""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any

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


class AngleStart(IntEnum):
    """Enumeration for angle starting positions."""

    EAST = 0  # 0 degrees
    NE = auto()  # 45 degrees
    NORTH = auto()  # 90 degrees
    NW = auto()  # 135 degrees
    WEST = auto()  # 180 degrees
    SW = auto()  # 225 degrees
    SOUTH = auto()  # 270 degrees
    SE = auto()  # 315 degrees


def octants(start: AngleStart | int = AngleStart.EAST) -> NDArray[Shape["8"], Float]:
    """Get octant angles starting from a specified position.

    Parameters
    ----------
    start
        Starting position for the octants (default is 90 degrees).

    Returns
    -------
    np.ndarray
        Array of octant angles in degrees starting from the specified position.

    Examples
    --------
    >>> octants(AngleStart.ONE)
    array([  0.,  45.,  90., 135., 180., 225., 270., 315.])
    >>> octants(AngleStart.FIVE)
    array([180., 225., 270., 315.,   0.,  45.,  90., 135.])

    """
    if isinstance(start, int):
        start = AngleStart(start)

    base_angles = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=float)
    start_index = start
    return np.roll(base_angles, -start_index)


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


def cosine_form(
    theta: NDArray[Shape[Any], Float], ampl: float, disp: float, elev: float
) -> NDArray[Shape[Any], Float]:
    """
    Cosine function with amplitude, displacement and elevation parameters.

    This is the mathematical model used in the Structural Summary Method.

    Parameters
    ----------
    theta
        Angular positions in radians.
    ampl
        Amplitude of the cosine curve.
    disp
        Angular displacement in radians.
    elev
        Elevation (mean level) of the cosine curve.

    Returns
    -------
    np.ndarray
        Predicted values at each theta position.
    """
    return elev + ampl * np.cos(theta - disp)


def sort_angles(
    angles: NDArray[Shape[Any], Float], scores: NDArray[Shape[Any], Float]
) -> tuple[NDArray[Shape[Any], Float], NDArray[Shape[Any], Float]]:
    """Sort angles and corresponding scores in ascending order."""
    sorted_indices = np.argsort(angles)
    return np.array(angles)[sorted_indices], np.array(scores)[sorted_indices]
