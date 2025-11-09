"""Score calculation functions for SSM analysis.

This module implements the core score calculation functions for both mean-based
and correlation-based SSM analysis, ported from the C++ implementation in the
R circumplex package.
"""

import numpy as np

from circumplex.core.parameters import ssm_parameters


def mean_scores(
    data: np.ndarray,
    groups: np.ndarray | None = None,
    *,
    listwise: bool = True,
) -> np.ndarray:
    """Calculate mean scale scores by group.

    Port of mean_scores() from R circumplex C++ implementation. Computes
    mean values for each scale, optionally stratified by group, with
    listwise or pairwise deletion of missing data.

    Parameters
    ----------
    data
        Array of circumplex scale scores, shape (n_obs, n_scales)
    groups
        Group indicators as integers (0-indexed), shape (n_obs,).
        If None, treats all observations as a single group.
    listwise
        If True, use listwise deletion (remove any row with any NA).
        If False, use pairwise deletion (compute mean per scale ignoring NAs).

    Returns
    -------
    Array of mean scores, shape (n_groups, n_scales)

    Examples
    --------
    >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> mean_scores(data)
    array([[4., 5., 6.]])

    >>> groups = np.array([0, 0, 1])
    >>> mean_scores(data, groups)
    array([[2.5, 3.5, 4.5],
           [7. , 8. , 9. ]])

    Notes
    -----
    This function mirrors the behavior of mean_scores() in the R package's
    C++ code (src/parameters.cpp lines 62-93).

    """
    # Handle single group case
    if groups is None:
        groups = np.zeros(len(data), dtype=int)

    # Get unique groups (sorted)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    n_scales = data.shape[1]

    # Initialize output
    result = np.zeros((n_groups, n_scales))

    for i, group_id in enumerate(unique_groups):
        # Extract data for this group
        group_mask = groups == group_id
        group_data = data[group_mask]

        if listwise:
            # Listwise deletion: remove rows with any NA
            complete_rows = ~np.isnan(group_data).any(axis=1)
            clean_data = group_data[complete_rows]
            if len(clean_data) > 0:
                result[i] = np.mean(clean_data, axis=0)
            else:
                result[i] = np.nan
        else:
            # Pairwise deletion: compute mean per column ignoring NAs
            result[i] = np.nanmean(group_data, axis=0)

    return result


def corr_scores(
    scales: np.ndarray,
    measures: np.ndarray,
    groups: np.ndarray | None = None,
    *,
    listwise: bool = True,
) -> np.ndarray:
    """Calculate correlation scores between measures and scales by group.

    Port of corr_scores() from R circumplex C++ implementation. Computes
    correlations between measure variables and circumplex scales, optionally
    stratified by group, with listwise or pairwise deletion.

    Parameters
    ----------
    scales
        Array of circumplex scale scores, shape (n_obs, n_scales)
    measures
        Array of measure variables, shape (n_obs, n_measures)
    groups
        Group indicators as integers (0-indexed), shape (n_obs,).
        If None, treats all observations as a single group.
    listwise
        If True, use listwise deletion (remove any row with any NA).
        If False, use pairwise deletion (compute correlation per pair ignoring NAs).

    Returns
    -------
    Array of correlation scores, shape (n_groups * n_measures, n_scales).
    Rows are ordered by group, then by measure within each group.

    Examples
    --------
    >>> scales = np.array([[1, 2], [3, 4], [5, 6]])
    >>> measures = np.array([[0], [1], [2]])
    >>> corr_scores(scales, measures)
    array([[1., 1.]])

    Notes
    -----
    This function mirrors the behavior of corr_scores() in the R package's
    C++ code (src/parameters.cpp lines 113-160).

    The output is organized as:
    - Single group: [measure1_corrs, measure2_corrs, ...]
    - Multiple groups: [g1_m1_corrs, g1_m2_corrs, ..., g2_m1_corrs, g2_m2_corrs, ...]

    """
    # Ensure measures is 2D
    if measures.ndim == 1:
        measures = measures.reshape(-1, 1)

    # Handle single group case
    if groups is None:
        groups = np.zeros(len(scales), dtype=int)

    # Get dimensions
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    n_measures = measures.shape[1]
    n_scales = scales.shape[1]

    # Initialize output: (n_groups * n_measures) x n_scales
    result = np.zeros((n_groups * n_measures, n_scales))

    for g_idx, group_id in enumerate(unique_groups):
        # Extract data for this group
        group_mask = groups == group_id
        group_scales = scales[group_mask]
        group_measures = measures[group_mask]

        if listwise:
            # Listwise deletion: remove rows with any NA in scales or measures
            all_data = np.column_stack([group_scales, group_measures])
            complete_rows = ~np.isnan(all_data).any(axis=1)
            clean_scales = group_scales[complete_rows]
            clean_measures = group_measures[complete_rows]

            # Compute correlations for all measure-scale pairs
            for m_idx in range(n_measures):
                row_idx = g_idx * n_measures + m_idx
                if len(clean_scales) > 1:  # Need at least 2 observations
                    for s_idx in range(n_scales):
                        result[row_idx, s_idx] = np.corrcoef(
                            clean_measures[:, m_idx], clean_scales[:, s_idx]
                        )[0, 1]
                else:
                    result[row_idx] = np.nan
        else:
            # Pairwise deletion: compute correlation per pair ignoring NAs
            for m_idx in range(n_measures):
                row_idx = g_idx * n_measures + m_idx
                for s_idx in range(n_scales):
                    # Get measure and scale vectors
                    m_vec = group_measures[:, m_idx]
                    s_vec = group_scales[:, s_idx]

                    # Remove pairs where either is NA
                    valid_mask = ~(np.isnan(m_vec) | np.isnan(s_vec))
                    m_clean = m_vec[valid_mask]
                    s_clean = s_vec[valid_mask]

                    if len(m_clean) > 1:  # Need at least 2 observations
                        result[row_idx, s_idx] = np.corrcoef(m_clean, s_clean)[0, 1]
                    else:
                        result[row_idx, s_idx] = np.nan

    return result


def group_parameters(
    scores: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    """Calculate SSM parameters for multiple groups.

    Applies ssm_parameters() to each row of a score matrix, returning
    a flat array of all parameters.

    Parameters
    ----------
    scores
        Array of scale scores, shape (n_groups, n_scales)
    angles
        Angular positions in radians, shape (n_scales,)

    Returns
    -------
    Flat array of parameters, length (n_groups * 6).
    Order: [e1, x1, y1, a1, d1, f1, e2, x2, y2, a2, d2, f2, ...]

    Examples
    --------
    >>> from circumplex.core.parameters import ssm_parameters
    >>> from circumplex.utils.angles import OCTANTS, degrees_to_radians
    >>> scores = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
    ...                     [8, 7, 6, 5, 4, 3, 2, 1]])
    >>> angles = degrees_to_radians(OCTANTS)
    >>> params = group_parameters(scores, angles)
    >>> len(params)
    12

    Notes
    -----
    This function mirrors group_parameters() in the R package's C++ code
    (src/parameters.cpp lines 37-45).

    """
    n_groups = scores.shape[0]
    result = np.zeros(n_groups * 6)

    for i in range(n_groups):
        params = ssm_parameters(scores[i], angles)
        result[i * 6 : (i + 1) * 6] = [
            params["elevation"],
            params["x_value"],
            params["y_value"],
            params["amplitude"],
            params["displacement"],
            params["fit"],
        ]

    return result
