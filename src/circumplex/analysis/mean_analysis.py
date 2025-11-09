"""Mean-based SSM analysis.

This module implements mean-based SSM analysis with bootstrap confidence
intervals, supporting single-group and multi-group designs with optional
contrast analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from circumplex.analysis.bootstrap import calculate_confidence_intervals, ssm_bootstrap
from circumplex.core.scores import group_parameters, mean_scores
from circumplex.utils.contrast import param_diff_array


def ssm_analyze_means(  # noqa: PLR0915
    data: pd.DataFrame,
    scales: list[str] | list[int],
    angles: np.ndarray,
    grouping: str | None = None,
    boots: int = 2000,
    interval: float = 0.95,
    *,
    contrast: bool = False,
    listwise: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """Perform mean-based SSM analysis.

    Calculates SSM parameters from mean scale scores, optionally stratified
    by group, with bootstrap confidence intervals. Supports contrast analysis
    for comparing two groups.

    Parameters
    ----------
    data
        DataFrame containing circumplex scale scores
    scales
        Column names or indices for circumplex scales (length n_scales)
    angles
        Angular positions in radians (length n_scales)
    grouping
        Column name for grouping variable. If None, analyzes all data as one group.
    contrast
        If True, calculate difference between two groups (requires exactly 2 groups)
    boots
        Number of bootstrap resamples
    interval
        Confidence level (e.g., 0.95 for 95% CI)
    listwise
        If True, use listwise deletion. If False, use pairwise deletion.
    seed
        Random seed for reproducibility

    Returns
    -------
    Dictionary with keys:
    - 'results': DataFrame with parameters and confidence intervals
    - 'scores': DataFrame with mean scale scores per group
    - 'details': Dict with analysis metadata
    - 'type': 'mean'

    Raises
    ------
    ValueError
        If contrast=True but number of groups is not 2

    Examples
    --------
    >>> from circumplex.data import load_dataset
    >>> from circumplex.utils.angles import OCTANTS, degrees_to_radians
    >>> data = load_dataset('jz2017')
    >>> angles = degrees_to_radians(OCTANTS)
    >>> results = ssm_analyze_means(data, scales=['PA', 'BC', 'DE', 'FG',
    ...                                             'HI', 'JK', 'LM', 'NO'],
    ...                              angles=angles, boots=2000, seed=12345)

    Notes
    -----
    This function mirrors ssm_analyze_means() from the R package
    (R/ssm_analysis.R lines 179-276).

    """
    # Convert scale indices to names if needed
    if isinstance(scales[0], int):
        scale_names = [data.columns[i] for i in scales]
    else:
        scale_names = scales

    # Handle grouping
    if grouping is None:
        # Create synthetic "All" group
        group_labels = np.array(["All"] * len(data))
        group_col = "All"
        n_groups = 1
    else:
        group_col = data[grouping]
        # Convert to categorical and get labels
        if not isinstance(group_col.dtype, pd.CategoricalDtype):
            group_col = pd.Categorical(group_col)
        else:
            group_col = group_col.astype("category")

        group_labels = np.array(group_col)
        unique_groups = group_col.categories
        n_groups = len(unique_groups)

    # Validate contrast
    if contrast and n_groups != 2:
        msg = "Contrast can only be TRUE when comparing exactly 2 groups"
        raise ValueError(msg)

    # Prepare bootstrap input data
    bs_input = data[scale_names].copy()
    bs_input["__group__"] = group_labels

    # Apply listwise deletion if requested
    if listwise:
        bs_input = bs_input.dropna()

    # Convert groups to integer codes for internal use
    if grouping is None:
        group_codes = np.zeros(len(bs_input), dtype=int)
    else:
        group_categories = pd.Categorical(bs_input["__group__"])
        group_codes = group_categories.codes
        unique_groups = group_categories.categories

    # Calculate observed mean scores
    observed_scores = mean_scores(
        bs_input[scale_names].values, group_codes, listwise=listwise
    )

    # Add contrast row if requested
    if contrast:
        contrast_scores = observed_scores[1] - observed_scores[0]
        observed_scores = np.vstack([observed_scores, contrast_scores])

    # Define bootstrap function
    def bootstrap_fn(df: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
        """Calculate SSM parameters for a bootstrap resample."""
        resampled = df.iloc[indices]
        scale_vals = resampled[scale_names].to_numpy()

        # Get group codes for resample
        if grouping is None:
            grp_codes = np.zeros(len(resampled), dtype=int)
        else:
            grp_categories = pd.Categorical(
                resampled["__group__"], categories=unique_groups
            )
            grp_codes = grp_categories.codes

        # Calculate mean scores for this resample
        scores_r = mean_scores(scale_vals, grp_codes, listwise=listwise)

        # Calculate SSM parameters
        params = group_parameters(scores_r, angles)

        # Add contrast if requested
        if contrast:
            contrast_params = param_diff_array(params)
            params = np.concatenate([params, contrast_params])

        return params

    # Perform bootstrap
    bs_results = ssm_bootstrap(
        bs_input,
        bootstrap_fn,
        boots=boots,
        grouping_col="__group__" if grouping is not None else None,
        seed=seed,
    )

    # Calculate confidence intervals
    results_df = calculate_confidence_intervals(bs_results)

    # Convert displacement from radians to degrees
    results_df["d_est"] = np.degrees(results_df["d_est"])
    if "d_lci" in results_df.columns:
        results_df["d_lci"] = np.degrees(results_df["d_lci"])
        results_df["d_uci"] = np.degrees(results_df["d_uci"])

    # Add metadata columns
    if grouping is None:
        results_df.insert(0, "Label", ["All"])
        group_labels_list = ["All"]
    else:
        group_labels_list = list(unique_groups)
        if contrast:
            contrast_label = f"{group_labels_list[1]} - {group_labels_list[0]}"
            group_labels_list.append(contrast_label)
        results_df.insert(0, "Label", group_labels_list)

    results_df.insert(1, "Group", group_labels_list)
    results_df.insert(2, "Measure", [None] * len(results_df))

    # Create scores DataFrame
    scores_df = pd.DataFrame(observed_scores, columns=scale_names)
    scores_df.insert(0, "Label", group_labels_list)

    # Prepare details
    details = {
        "boots": boots,
        "interval": interval,
        "listwise": listwise,
        "angles": np.degrees(angles).tolist(),
        "contrast": contrast,
        "score_type": "mean",
    }

    return {
        "results": results_df,
        "scores": scores_df,
        "details": details,
        "type": "mean",
    }
