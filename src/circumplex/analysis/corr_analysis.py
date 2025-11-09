"""Correlation-based SSM analysis.

This module implements correlation-based SSM analysis with bootstrap confidence
intervals, supporting single/multi-group and single/multi-measure designs with
optional contrast analysis.
"""

from typing import Any

import numpy as np
import pandas as pd

from circumplex.analysis.bootstrap import calculate_confidence_intervals, ssm_bootstrap
from circumplex.core.scores import corr_scores, group_parameters
from circumplex.utils.contrast import param_diff_array


def ssm_analyze_corrs(  # noqa: C901, PLR0915
    data: pd.DataFrame,
    scales: list[str] | list[int],
    angles: np.ndarray,
    measures: list[str] | str,
    grouping: str | None = None,
    boots: int = 2000,
    interval: float = 0.95,
    measures_labels: list[str] | None = None,
    *,
    contrast: bool = False,
    listwise: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """Perform correlation-based SSM analysis.

    Calculates SSM parameters from correlations between measures and scales,
    optionally stratified by group, with bootstrap confidence intervals.
    Supports contrast analysis for comparing two groups or two measures.

    Parameters
    ----------
    data
        DataFrame containing circumplex scales and measures
    scales
        Column names or indices for circumplex scales (length n_scales)
    angles
        Angular positions in radians (length n_scales)
    measures
        Column name(s) for measure variable(s). Can be string or list.
    grouping
        Column name for grouping variable. If None, analyzes all data as one group.
    boots
        Number of bootstrap resamples
    interval
        Confidence level (e.g., 0.95 for 95% CI)
    measures_labels
        Optional custom labels for measures (same length as measures)
    contrast
        If True, calculate difference between two groups or two measures
        (requires exactly 2 groups OR 2 measures, not both)
    listwise
        If True, use listwise deletion. If False, use pairwise deletion.
    seed
        Random seed for reproducibility

    Returns
    -------
    Dictionary with keys:
    - 'results': DataFrame with parameters and confidence intervals
    - 'scores': DataFrame with correlation scores
    - 'details': Dict with analysis metadata
    - 'type': 'correlation'

    Raises
    ------
    ValueError
        If contrast=True but requirements not met (2 groups XOR 2 measures)

    Examples
    --------
    >>> from circumplex.data import load_dataset
    >>> from circumplex.utils.angles import OCTANTS, degrees_to_radians
    >>> data = load_dataset('jz2017')
    >>> angles = degrees_to_radians(OCTANTS)
    >>> results = ssm_analyze_corrs(data, scales=['PA', 'BC', 'DE', 'FG',
    ...                                             'HI', 'JK', 'LM', 'NO'],
    ...                              angles=angles, measures='PARPD',
    ...                              boots=2000, seed=12345)

    Notes
    -----
    This function mirrors ssm_analyze_corrs() from the R package
    (R/ssm_analysis.R lines 280-406).

    """
    # Convert measures to list if single string
    if isinstance(measures, str):
        measures = [measures]

    # Convert scale indices to names if needed
    if isinstance(scales[0], int):
        scale_names = [data.columns[i] for i in scales]
    else:
        scale_names = scales

    # Use custom labels or measure names
    if measures_labels is None:
        measures_labels = measures

    n_measures = len(measures)

    # Handle grouping
    if grouping is None:
        group_labels = np.array(["All"] * len(data))
        group_col = "All"
        n_groups = 1
    else:
        group_col = data[grouping]
        if not isinstance(group_col.dtype, pd.CategoricalDtype):
            group_col = pd.Categorical(group_col)
        else:
            group_col = group_col.astype("category")

        group_labels = np.array(group_col)
        unique_groups = group_col.categories
        n_groups = len(unique_groups)

    # Validate contrast
    if contrast:
        group_mean_contrast = n_measures == 0 and n_groups == 2
        group_corr_contrast = n_measures == 1 and n_groups == 2
        measure_corr_contrast = n_measures == 2 and n_groups == 1

        if not (group_mean_contrast or group_corr_contrast or measure_corr_contrast):
            msg = (
                "Contrast can only be TRUE when comparing exactly 2 groups "
                "(with 1 measure) or exactly 2 measures (with 1 group)"
            )
            raise ValueError(msg)

    # Prepare bootstrap input data
    bs_input = data[scale_names + measures].copy()
    bs_input["__group__"] = group_labels

    # Apply listwise deletion if requested
    if listwise:
        bs_input = bs_input.dropna()

    # Convert groups to integer codes
    if grouping is None:
        group_codes = np.zeros(len(bs_input), dtype=int)
        unique_groups = ["All"]
    else:
        group_categories = pd.Categorical(bs_input["__group__"])
        group_codes = group_categories.codes
        unique_groups = group_categories.categories

    # Calculate observed correlation scores
    observed_scores = corr_scores(
        bs_input[scale_names].values,
        bs_input[measures].values,
        group_codes,
        listwise=listwise,
    )

    # Add contrast row if requested
    if contrast:
        if n_measures == 2 and n_groups == 1:
            # Measure contrast: compare two measures within single group
            contrast_scores = observed_scores[1] - observed_scores[0]
        elif n_groups == 2:
            # Group contrast: compare two groups for single measure
            contrast_scores = observed_scores[1] - observed_scores[0]
        observed_scores = np.vstack([observed_scores, contrast_scores])

    # Define bootstrap function
    def bootstrap_fn(df: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
        """Calculate SSM parameters for a bootstrap resample."""
        resampled = df.iloc[indices]
        scale_vals = resampled[scale_names].to_numpy()
        measure_vals = resampled[measures].to_numpy()

        # Get group codes for resample
        if grouping is None:
            grp_codes = np.zeros(len(resampled), dtype=int)
        else:
            grp_categories = pd.Categorical(
                resampled["__group__"], categories=unique_groups
            )
            grp_codes = grp_categories.codes

        # Calculate correlation scores for this resample
        scores_r = corr_scores(scale_vals, measure_vals, grp_codes, listwise=listwise)

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

    # Create labels combining measure and group information
    labels = []
    profile_groups = []
    profile_measures = []

    if n_groups == 1:
        # Single group: labels are measure names
        for m_label in measures_labels:
            labels.append(m_label)
            profile_groups.append(unique_groups[0])
            profile_measures.append(m_label)

        if contrast and n_measures == 2:
            contrast_label = f"{measures_labels[1]} - {measures_labels[0]}"
            labels.append(contrast_label)
            profile_groups.append(unique_groups[0])
            profile_measures.append(contrast_label)
    else:
        # Multiple groups: labels combine measure and group
        for group in unique_groups:
            for m_label in measures_labels:
                # Always use "MEASURE: GROUP" format for multi-group correlation
                label = f"{m_label}: {group}"
                labels.append(label)
                profile_groups.append(group)
                profile_measures.append(m_label)

        if contrast:
            if n_measures == 1:
                # Format: "MEASURE: GROUP1 - GROUP2"  # noqa: ERA001
                contrast_label = (
                    f"{measures_labels[0]}: {unique_groups[1]} - {unique_groups[0]}"
                )
            else:
                contrast_label = (
                    f"{unique_groups[1]} - {unique_groups[0]}: {measures_labels[0]}"
                )
            labels.append(contrast_label)
            profile_groups.append(contrast_label)
            profile_measures.append(
                measures_labels[0] if n_measures == 1 else contrast_label
            )

    # Add metadata columns
    results_df.insert(0, "Label", labels)
    results_df.insert(1, "Group", profile_groups)
    results_df.insert(2, "Measure", profile_measures)

    # Create scores DataFrame
    scores_df = pd.DataFrame(observed_scores, columns=scale_names)
    scores_df.insert(0, "Label", labels)

    # Prepare details
    details = {
        "boots": boots,
        "interval": interval,
        "listwise": listwise,
        "angles": np.degrees(angles).tolist(),
        "contrast": contrast,
        "score_type": "correlation",
    }

    return {
        "results": results_df,
        "scores": scores_df,
        "details": details,
        "type": "correlation",
    }
