"""Main SSM analysis function.

This module provides the primary user-facing function for performing
Structural Summary Method (SSM) analysis on circumplex data.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from circumplex.analysis.corr_analysis import ssm_analyze_corrs
from circumplex.analysis.mean_analysis import ssm_analyze_means
from circumplex.ssm import SSM
from circumplex.utils.angles import OCTANTS, degrees_to_radians


def ssm_analyze(
    data: pd.DataFrame,
    scales: list[str] | list[int],
    angles: np.ndarray | list[float] | None = None,
    measures: list[str] | str | None = None,
    grouping: str | None = None,
    boots: int = 2000,
    interval: float = 0.95,
    measures_labels: list[str] | None = None,
    *,
    contrast: bool = False,
    listwise: bool = True,
    seed: int | None = None,
) -> SSM:
    """Perform Structural Summary Method (SSM) analysis on circumplex data.

    This is the main entry point for SSM analysis. It automatically determines
    whether to perform mean-based or correlation-based analysis based on the
    `measures` parameter, calculates SSM parameters, and computes bootstrap
    confidence intervals.

    Parameters
    ----------
    data
        DataFrame containing circumplex scale scores (and optionally measures
        and grouping variables)
    scales
        Column names or indices for circumplex scales. Must be ordered according
        to their angular positions (matching the order in `angles`).
    angles
        Angular positions for the scales in degrees. If None, uses standard
        octant angles [90, 135, 180, 225, 270, 315, 360, 45]. Length must
        match the number of scales.
    measures
        Column name(s) for external measure variable(s) to correlate with scales.
        - If None: Performs mean-based analysis (profile analysis)
        - If string or list: Performs correlation-based analysis
    grouping
        Column name for grouping variable. If None, treats all data as a
        single group. The variable will be converted to a categorical factor.
    contrast
        If True, calculates differences between groups or measures.
        - Requires exactly 2 groups (with 0 or 1 measures), OR
        - Requires exactly 2 measures (with 1 group)
        Raises ValueError if requirements not met.
    boots
        Number of bootstrap resamples for confidence interval calculation.
        Default: 2000.
    interval
        Confidence level for bootstrap intervals (e.g., 0.95 for 95% CI).
        Default: 0.95.
    listwise
        Missing data handling:
        - If True: Listwise deletion (remove rows with any missing value)
        - If False: Pairwise deletion (use all available data for each calculation)
        Default: True.
    measures_labels
        Optional custom labels for measures (same length as measures).
        If None, uses the column names.
    seed
        Random seed for reproducibility of bootstrap results. If None, results
        will vary across runs.

    Returns
    -------
    dict
        Dictionary containing:

        - **results** : pd.DataFrame
            SSM parameters and confidence intervals with columns:

            - Label: Profile label (group/measure name or combination)
            - Group: Group identifier
            - Measure: Measure identifier (None for mean-based analysis)
            - e_est, x_est, y_est, a_est, d_est, fit_est: Point estimates
            - e_lci, x_lci, y_lci, a_lci, d_lci: Lower confidence intervals
            - e_uci, x_uci, y_uci, a_uci, d_uci: Upper confidence intervals

            Note: Displacement (d) is in degrees. No CIs for fit parameter.

        - **scores** : pd.DataFrame
            Mean scores (mean-based) or correlation scores (correlation-based)
            for each scale, with Label column.

        - **details** : dict
            Analysis metadata including boots, interval, listwise, angles,
            contrast, and score_type.

        - **type** : str
            Analysis type: 'mean' or 'correlation'

    Raises
    ------
    ValueError
        - If angles length doesn't match scales length
        - If contrast=True but requirements not met
        - If data contains no valid observations after missing data handling

    Examples
    --------
    **Mean-based analysis (single group)**

    >>> from circumplex import ssm_analyze
    >>> from circumplex.data import load_dataset
    >>> aw2009 = load_dataset('aw2009')
    >>> results = ssm_analyze(aw2009, scales=list(range(8)), seed=12345)
    >>> print(results['results'][['Label', 'e_est', 'a_est', 'd_est']])
         Label  e_est  a_est      d_est
    0      All  0.423  0.981  344.358

    **Mean-based analysis (multiple groups with contrast)**

    >>> jz2017 = load_dataset('jz2017')
    >>> results = ssm_analyze(jz2017, scales=list(range(1, 9)),
    ...                        grouping='Gender', contrast=True, seed=12345)
    >>> print(results['results'][['Label', 'e_est', 'a_est']])
              Label  e_est  a_est
    0        Female  0.635  0.158
    1          Male  0.596  0.192
    2  Male - Female -0.039  0.034

    **Correlation-based analysis (single measure)**

    >>> results = ssm_analyze(jz2017, scales=list(range(1, 9)),
    ...                        measures='PARPD', seed=12345)
    >>> print(results['results'][['Label', 'e_est', 'a_est', 'd_est']])
       Label  e_est  a_est   d_est
    0  PARPD  0.250  0.150  128.9

    **Correlation-based analysis (measure contrast)**

    >>> results = ssm_analyze(jz2017, scales=list(range(1, 9)),
    ...                        measures=['ASPD', 'NARPD'],
    ...                        contrast=True, seed=12345)
    >>> print(results['results'][['Label', 'e_est', 'a_est']])
                 Label  e_est  a_est
    0             ASPD  0.253  0.055
    1            NARPD  0.311  0.203
    2  NARPD - ASPD     0.058  0.148

    Notes
    -----
    This function is a Python port of ssm_analyze() from the R circumplex
    package (Zimmermann & Wright, 2017). It maintains numerical parity with
    the R implementation to at least 3 decimal places.

    **SSM Parameters:**

    - **elevation (e)**: Mean of all scale scores
    - **x_value (x)**: Projection onto x-axis (cosine component)
    - **y_value (y)**: Projection onto y-axis (sine component)
    - **amplitude (a)**: Vector length (prototypicality)
    - **displacement (d)**: Angular position in degrees [0, 360)
    - **fit**: Model fit (R²), proportion of variance explained

    **Bootstrap Confidence Intervals:**

    Uses percentile method with stratified sampling when groups are present.
    Displacement CIs use circular statistics to handle angular wrapping.

    See Also
    --------
    load_dataset : Load example datasets
    OCTANTS : Standard octant angles for 8-scale circumplex

    References
    ----------
    Zimmermann, J., & Wright, A. G. C. (2017). Beyond description in
    interpersonal construct validation: Methodological advances in the
    circumplex Structural Summary Approach. *Assessment, 24*(1), 3-23.
    https://doi.org/10.1177/1073191115621795

    Zimmermann, J., & Wright, A. G. C. (2017). The circumplex package
    [Computer software]. https://cran.r-project.org/package=circumplex

    """
    # Validate and process scales
    if isinstance(scales[0], int):
        scale_names = [data.columns[i] for i in scales]
    else:
        scale_names = scales

    n_scales = len(scale_names)

    # Process angles
    if angles is None:
        # Use default octant angles
        if n_scales != 8:
            msg = (
                f"When angles=None, exactly 8 scales are required (got {n_scales}). "
                "Please provide custom angles for non-octant circumplex models."
            )
            raise ValueError(msg)
        angles_deg = OCTANTS
    else:
        # Convert to numpy array if needed
        angles_deg = np.array(angles, dtype=float)

        # Validate length
        if len(angles_deg) != n_scales:
            msg = (
                f"Length of angles ({len(angles_deg)}) must match "
                f"number of scales ({n_scales})"
            )
            raise ValueError(msg)

    # Convert angles to radians for internal use
    # Ensure numpy array type for downstream functions
    angles_rad = cast("np.ndarray", degrees_to_radians(angles_deg))

    # Route to appropriate analysis function
    if measures is None:
        # Mean-based analysis
        return SSM.from_dict(
            ssm_analyze_means(
                data=data,
                scales=scale_names,
                angles=angles_rad,
                grouping=grouping,
                contrast=contrast,
                boots=boots,
                interval=interval,
                listwise=listwise,
                seed=seed,
            )
        )
    # Correlation-based analysis
    return SSM.from_dict(
        ssm_analyze_corrs(
            data=data,
            scales=scale_names,
            angles=angles_rad,
            measures=measures,
            grouping=grouping,
            contrast=contrast,
            boots=boots,
            interval=interval,
            listwise=listwise,
            measures_labels=measures_labels,
            seed=seed,
        )
    )
