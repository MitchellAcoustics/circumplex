"""SSM parameter calculation engine.

This module implements the core Structural Summary Method parameter
calculations, ported from the C++ implementation in the R package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from typing import Any

    from nptyping import NDArray, Shape


def ssm_parameters(
    scores: NDArray[Shape[Any], float],
    angles: NDArray[Shape[Any], float],
) -> dict[str, float]:
    """Calculate SSM parameters for a profile of circumplex scores.

    Direct port of ssm_parameters_cpp from the R circumplex package.
    Computes the six core SSM parameters: elevation, x-value, y-value,
    amplitude, displacement, and model fit.

    Parameters
    ----------
    scores
        Array of circumplex scale scores (length n_scales)
    angles
        Angular positions in radians (length n_scales)

    Returns
    -------
    Dictionary with keys:

    - elevation: Mean of all scale scores
    - x_value: Projection onto x-axis (cosine component)
    - y_value: Projection onto y-axis (sine component)
    - amplitude: Vector length in 2D space
    - displacement: Angular position in radians [0, 2π)
    - fit: Model fit (R²), proportion of variance explained

    Examples
    --------
    >>> import numpy as np
    >>> from circumplex.utils.angles import OCTANTS, degrees_to_radians
    >>>
    >>> scores = np.array([0.374, -0.572, -0.520, 0.016, 0.688, 1.142, 1.578, 0.678])
    >>> angles = degrees_to_radians(OCTANTS)
    >>> params = ssm_parameters(scores, angles)
    >>> print(f"Elevation: {params['elevation']:.3f}")
    Elevation: 0.423
    >>> print(f"Amplitude: {params['amplitude']:.3f}")
    Amplitude: 0.981

    Notes
    -----
    This function implements the Structural Summary Method as described in
    Zimmermann & Wright (2017). The displacement is returned in radians;
    convert to degrees if needed.

    References
    ----------
    Zimmermann, J., & Wright, A. G. C. (2017). Beyond description in
    interpersonal construct validation: Methodological advances in the
    circumplex Structural Summary Approach. Assessment, 24(1), 3-23.

    """
    n = len(scores)

    # Elevation: mean of scores
    elevation: float = cast("float", np.mean(scores))

    # X and Y values: projections onto axes
    x_value: float = (2 / n) * np.dot(scores, np.cos(angles))
    y_value: float = (2 / n) * np.dot(scores, np.sin(angles))

    # Amplitude: vector length
    amplitude: float = np.sqrt(x_value**2 + y_value**2)

    # Displacement: angular position
    # Use atan2 for proper quadrant, then modulo to ensure [0, 2π)
    displacement: float = np.arctan2(y_value, x_value) % (2 * np.pi)

    # Model fit (R²)
    # Predicted scores from cosine model
    predicted: NDArray[Any, float] = elevation + amplitude * np.cos(
        angles - displacement
    )
    ss_residual: float = np.sum((scores - predicted) ** 2)
    ss_total: float = cast("float", np.var(scores, ddof=1) * (n - 1))

    # Avoid division by zero
    fit: float = 1.0 if ss_total == 0 else 1 - ss_residual / ss_total

    return {
        "elevation": elevation,
        "x_value": x_value,
        "y_value": y_value,
        "amplitude": amplitude,
        "displacement": displacement,
        "fit": fit,
    }
