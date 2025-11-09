"""Bootstrap confidence interval calculation for SSM analysis.

This module implements bootstrap resampling with confidence interval
calculation, including special handling for circular displacement data.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from nptyping import Float, NDArray, Shape


def circular_quantile(
    angles: NDArray[Any, Float],
    probs: list[float] | NDArray[Shape[Any], Float],
) -> NDArray[Shape[Any], Float]:
    """Calculate quantiles for circular data in radians.

    Implements a circular quantile method that accounts for the periodic
    nature of angular data. Centers angles around their mean direction,
    calculates linear quantiles, then transforms back.

    Parameters
    ----------
    angles
        Array of angles in radians, shape (n,)
    probs
        Probability points at which to calculate quantiles (e.g., [0.025, 0.975])

    Returns
    -------
    Quantiles at the requested probability points

    Examples
    --------
    >>> angles = np.array([0.1, 0.2, 6.2, 6.3])  # Two near 0, two near 2π
    >>> circular_quantile(angles, [0.25, 0.75])
    array([6.25..., 0.15...])

    Notes
    -----
    This function mirrors the quantile.circumplex_radian method from the
    R package (R/ssm_bootstrap.R lines 72-82). It:
    1. Computes mean direction using atan2
    2. Centers all angles around the mean
    3. Calculates linear quantiles on centered data
    4. Transforms back to [0, 2π)

    """
    # Calculate mean direction
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))

    # Center angles around mean direction
    angles_centered = (angles - mean_angle + np.pi) % (2 * np.pi) - np.pi

    # Calculate quantiles on centered data
    quantiles_centered = np.quantile(angles_centered, probs)

    # Transform back
    return (quantiles_centered + mean_angle) % (2 * np.pi)


def ssm_bootstrap(
    data: pd.DataFrame,
    bootstrap_fn: Callable[
        [pd.DataFrame, NDArray[Shape[Any], Float]], NDArray[Shape[Any], Float]
    ],
    boots: int = 2000,
    grouping_col: str | None = None,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Perform stratified bootstrap with confidence intervals.

    Executes bootstrap resampling with stratification by group (if specified),
    calculates point estimates and and bootstrap replicats;
    use `calculate_confidence_intervals()` to derive confidence intervals.

    Parameters
    ----------
    data
        DataFrame containing all data for bootstrap sampling
    bootstrap_fn
        Function that takes (data, resample_indices) and returns
        flat array of parameters for all groups/measures
    boots
        Number of bootstrap resamples
    grouping_col
        Name of grouping column for stratified sampling.
        If None, uses simple random sampling.
    seed
        Random seed for reproducibility

    Returns
    -------
    Dictionary containing:
    - 't0': Point estimates (observed parameters)
    - 't': Bootstrap matrix (boots x n_params)
    - 'n_params': Number of parameters per profile
    - 'n_profiles': Number of profiles (groups/measures)

    Examples
    --------
    >>> def simple_mean(df, indices):
    ...     return np.array([df.iloc[indices]['x'].mean()])
    >>> data = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
    >>> result = ssm_bootstrap(data, simple_mean, boots=100, seed=123)
    >>> result['t0']  # Observed mean
    array([3.])

    Notes
    -----
    This function mirrors ssm_bootstrap() from the R package
    (R/ssm_bootstrap.R lines 1-55). It uses stratified sampling
    when a grouping variable is provided to ensure each bootstrap
    sample maintains the original group proportions.

    """
    if seed is not None:
        np.random.seed(seed)

    n_obs = len(data)

    # Calculate observed parameters (t0)
    observed_indices = np.arange(n_obs)
    t0 = bootstrap_fn(data, observed_indices)

    # Initialize bootstrap matrix
    n_params_total = len(t0)
    t_matrix = np.zeros((boots, n_params_total))

    # Perform bootstrap resampling
    for b in range(boots):
        if grouping_col is not None:
            # Stratified sampling: sample within each group
            resample_indices = _stratified_resample(data, grouping_col)
        else:
            # Simple random sampling with replacement
            resample_indices = np.random.choice(n_obs, size=n_obs, replace=True)

        # Calculate parameters for this resample
        t_matrix[b] = bootstrap_fn(data, resample_indices)

    return {
        "t0": t0,
        "t": t_matrix,
        "n_params": 6,  # Always 6 SSM parameters per profile
        "n_profiles": n_params_total // 6,
    }


def _stratified_resample(data: pd.DataFrame, grouping_col: str) -> np.ndarray:
    """Perform stratified resampling within groups.

    Parameters
    ----------
    data
        DataFrame with grouping column
    grouping_col
        Name of column containing group labels

    Returns
    -------
    Array of resampled indices

    """
    indices = []
    groups = data[grouping_col].unique()

    for group in groups:
        # Get indices for this group
        group_mask = data[grouping_col] == group
        group_indices = np.where(group_mask)[0]

        # Sample with replacement from this group
        n_group = len(group_indices)
        resampled = np.random.choice(group_indices, size=n_group, replace=True)
        indices.extend(resampled)

    return np.array(indices)


def calculate_confidence_intervals(
    bootstrap_results: dict[str, Any],
    interval: float = 0.95,
) -> pd.DataFrame:
    """Calculate confidence intervals from bootstrap results.

    Computes percentile confidence intervals for all parameters, with
    special circular handling for displacement parameters.

    Parameters
    ----------
    bootstrap_results
        Dictionary from ssm_bootstrap() containing 't0' and 't'
    interval
        Confidence level (e.g., 0.95 for 95% CI)

    Returns
    -------
    DataFrame with columns:
    - e_est, x_est, y_est, a_est, d_est, fit_est (point estimates)
    - e_lci, x_lci, y_lci, a_lci, d_lci (lower CI)
    - e_uci, x_uci, y_uci, a_uci, d_uci (upper CI)
    Note: fit has no confidence intervals

    Notes
    -----
    This function mirrors the CI calculation in R's ssm_bootstrap()
    (R/ssm_bootstrap.R lines 38-54). It uses percentile method for
    linear parameters and circular_quantile() for displacement.

    """
    t0 = bootstrap_results["t0"]
    t_matrix = bootstrap_results["t"]
    n_params = bootstrap_results["n_params"]
    n_profiles = bootstrap_results["n_profiles"]

    # Calculate probability points for CI
    alpha = 1 - interval
    lower_prob = alpha / 2
    upper_prob = 1 - alpha / 2

    # Initialize results DataFrame
    param_names = ["e", "x", "y", "a", "d", "fit"]
    results = []

    for profile_idx in range(n_profiles):
        # Extract parameters for this profile
        param_start = profile_idx * n_params
        profile_params = {}

        for param_idx, param_name in enumerate(param_names):
            obs_value = t0[param_start + param_idx]
            boot_values = t_matrix[:, param_start + param_idx]

            # Point estimate
            profile_params[f"{param_name}_est"] = obs_value

            # Confidence intervals (skip fit)
            if param_name != "fit":
                if param_name == "d":
                    # Use circular quantile for displacement
                    ci = circular_quantile(boot_values, [lower_prob, upper_prob])

                else:
                    # Use regular quantile for other parameters
                    ci = np.quantile(boot_values, [lower_prob, upper_prob])

                profile_params[f"{param_name}_lci"] = ci[0]  # type: ignore[non-subscriptable]
                profile_params[f"{param_name}_uci"] = ci[1]  # type: ignore[non-subscriptable]

        results.append(profile_params)

    return pd.DataFrame(results)
