from __future__ import annotations

import warnings
from typing import Callable, List, Optional, Tuple, Union, Dict, Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import circumplex.ssm_results as ssm_results
import circumplex.utils as utils

if TYPE_CHECKING:
    from nptyping import NDArray

BOUNDS = ([0, -2 * np.pi, -np.inf], [np.inf, 2 * np.pi, np.inf])
OCTANTS = utils.OCTANTS


def validate_ssm_input(
    data: pd.DataFrame,
    scales: List[str],
    angles: Tuple[float, ...],
    measures: Optional[List[str]] = None,
    grouping: Optional[str] = None,
    contrast: str = "none",
    boots: int = 500,
    interval: float = 0.95,
    listwise: bool = True,
    measures_labels: Optional[List[str]] = None,
) -> None:
    """
    Validate input parameters for SSM analysis functions.

    Raises ValueError with helpful messages if validation fails.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing at least circumplex scales.
    scales : List[str]
        The variable names for the circumplex scales to be analyzed.
    angles : Tuple[float]
        Angular displacement of each circumplex scale (in degrees).
    measures : Optional[List[str]]
        Variables to be correlated with the circumplex scales.
    grouping : Optional[str]
        Variable name that indicates group membership of each observation.
    contrast : str
        Type of contrast to run ("none", "model", or "test").
    boots : int
        Number of bootstrap resamples for estimating confidence intervals.
    interval : float
        Confidence level for estimating the confidence intervals.
    listwise : bool
        Whether to use listwise deletion for missing values.
    measures_labels : Optional[List[str]]
        Labels for each measure provided in measures.

    Returns
    -------
    None
    """
    # Data validation
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")

    # Scales validation
    if not all(scale in data.columns for scale in scales):
        missing = set(scales) - set(data.columns)
        raise ValueError(f"Scales missing from data: {missing}")

    # Angles validation
    if not (
        isinstance(angles, tuple) and all(isinstance(a, (int, float)) for a in angles)
    ):
        raise ValueError("angles must be a tuple of numbers")
    if len(angles) != len(scales):
        raise ValueError(
            f"angles and scales must have the same length (angles: {len(angles)}, scales: {len(scales)})"
        )

    # Bootstrap validation
    if not (isinstance(boots, int) and boots > 0):
        raise ValueError("boots must be a positive integer")

    # Interval validation
    if not (0 < interval < 1):
        raise ValueError("interval must be between 0 and 1")

    # Listwise validation
    if not isinstance(listwise, bool):
        raise ValueError("listwise must be a boolean")

    # Contrast validation
    if contrast not in ["none", "model", "test"]:
        raise ValueError("contrast must be 'none', 'model', or 'test'")

    # Measures validation
    if measures is not None and not all(
        measure in data.columns for measure in measures
    ):
        missing = set(measures) - set(data.columns)
        raise ValueError(f"Measures missing from data: {missing}")

    # Grouping validation
    if grouping is not None and grouping not in data.columns:
        raise ValueError(f"grouping variable '{grouping}' not found in data")

    # Measures labels validation
    if measures_labels is not None:
        if measures is None:
            raise ValueError(
                "measures must be provided when measures_labels is provided"
            )
        if len(measures_labels) != len(measures):
            raise ValueError("measures_labels must have the same length as measures")

    # Contrast possibility validation
    if contrast != "none" and measures is None and grouping is None:
        raise ValueError(
            "Without specifying measures or grouping, no contrasts are possible. "
            "Set contrast = 'none' or add the measures or grouping arguments."
        )


def ssm_analyze(
    data: pd.DataFrame,
    scales: List[str],
    angles: Tuple[float, ...] = OCTANTS,
    measures: Optional[List[str]] = None,
    grouping: Optional[str] = None,
    contrast: str = "none",
    boots: int = 500,
    interval: float = 0.95,
    listwise: bool = True,
    measures_labels: Optional[List[str]] = None,
) -> ssm_results.SSMResults:
    """
    Perform analyses using the Structural Summary Method.

    This function calculates SSM parameters with bootstrapped confidence intervals for various
    analysis types. Depending on the arguments supplied, it performs either mean-based or
    correlation-based analyses, uses one or more groups to stratify the data, and calculates
    contrasts between groups or measures.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing at least circumplex scales.
    scales : List[str]
        The variable names for the circumplex scales to be analyzed.
    angles : Tuple[float], optional
        Angular displacement of each circumplex scale (in degrees).
        Default is (0, 45, 90, 135, 180, 225, 270, 315).
    measures : Optional[List[str]], optional
        Variables to be correlated with the circumplex scales.
    grouping : Optional[str], optional
        Variable name that indicates group membership of each observation.
    contrast : str, optional
        Type of contrast to run ("none", "model", or "test").
        Default is "none".
    boots : int, optional
        Number of bootstrap resamples for estimating confidence intervals.
        Default is 500.
    interval : float, optional
        Confidence level for estimating the confidence intervals.
        Default is 0.95.
    listwise : bool, optional
        Whether to use listwise deletion for missing values.
        Default is True.
    measures_labels : Optional[List[str]], optional
        Labels for each measure provided in measures.

    Returns
    -------
    SSMResults
        An object containing the results and description of the analysis.

    Examples
    --------
    >>> import pandas as pd
    >>> from circumplex import ssm_analyze
    >>>
    >>> # Simple analysis of means
    >>> results = ssm_analyze(
    ...     data,
    ...     scales=["PA", "BC", "DE", "FG", "HI", "JK", "LM", "NO"],
    ...     angles=(0, 45, 90, 135, 180, 225, 270, 315)
    ... )
    >>>
    >>> # Analysis with correlations and groups
    >>> results = ssm_analyze(
    ...     data,
    ...     scales=["PA", "BC", "DE", "FG", "HI", "JK", "LM", "NO"],
    ...     angles=(0, 45, 90, 135, 180, 225, 270, 315),
    ...     measures=["Extraversion", "Neuroticism"],
    ...     grouping="Gender"
    ... )
    """
    # Validate all input parameters
    validate_ssm_input(
        data=data,
        scales=scales,
        angles=angles,
        measures=measures,
        grouping=grouping,
        contrast=contrast,
        boots=boots,
        interval=interval,
        listwise=listwise,
        measures_labels=measures_labels,
    )

    # Determine analysis type and forward to appropriate subfunction
    if measures is not None:
        if grouping is not None:
            # Multiple group correlations
            results = ssm_analyze_corrs(
                data,
                scales,
                angles,
                measures,
                grouping,
                contrast,
                boots,
                interval,
                listwise,
                measures_labels,
            )
        else:
            # Single group correlations
            results = ssm_analyze_corrs(
                data,
                scales,
                angles,
                measures,
                None,
                contrast,
                boots,
                interval,
                listwise,
                measures_labels,
            )
    else:
        if grouping is not None:
            # Multiple group means
            results = ssm_analyze_means(
                data, scales, angles, grouping, contrast, boots, interval, listwise
            )
        else:
            # Single group means
            if contrast != "none":
                raise ValueError(
                    "Without specifying measures or grouping, no contrasts are possible. "
                    "Set contrast = 'none' or add the measures or grouping arguments."
                )
            results = ssm_analyze_means(
                data, scales, angles, None, "none", boots, interval, listwise
            )

    # Create the call string
    call_str = (
        f"ssm_analyze(data, scales={scales}, angles={angles}, measures={measures}, "
        f"grouping={grouping}, contrast={contrast}, boots={boots}, "
        f"interval={interval}, listwise={listwise}, measures_labels={measures_labels})"
    )

    # Create and return the SSMResults object
    return ssm_results.SSMResults(
        results=results["results"],
        scales=scales,
        scores=results["scores"],
        details=results["details"],
        call=call_str,
    )


def ssm_analyze_means(
    data: pd.DataFrame,
    scales: List[str],
    angles: Tuple[float, ...],
    grouping: Optional[str] = None,
    contrast: str = "none",
    boots: int = 2000,
    interval: float = 0.95,
    listwise: bool = True,
) -> Dict[str, Any]:
    """
    Perform analyses using the mean-based Structural Summary Method.

    This function calculates SSM parameters based on the mean scores of
    circumplex scales, optionally grouped by a categorical variable.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing at least circumplex scales.
    scales : List[str]
        The variable names for the circumplex scales to be analyzed.
    angles : Tuple[float]
        Angular displacement of each circumplex scale (in degrees).
    grouping : Optional[str], optional
        Variable name that indicates group membership of each observation.
    contrast : str, optional
        Type of contrast to run ("none", "model", or "test").
        Default is "none".
    boots : int, optional
        Number of bootstrap resamples for estimating confidence intervals.
        Default is 2000.
    interval : float, optional
        Confidence level for estimating the confidence intervals.
        Default is 0.95.
    listwise : bool, optional
        Whether to use listwise deletion for missing values.
        Default is True.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the results and description of the analysis:
        - results: DataFrame with SSM parameters for each group
        - details: Dictionary with analysis details
        - call: String representation of the function call
        - scores: DataFrame with mean scores for each scale and group

    Examples
    --------
    >>> results_dict = ssm_analyze_means(
    ...     data,
    ...     scales=["PA", "BC", "DE", "FG", "HI", "JK", "LM", "NO"],
    ...     angles=(0, 45, 90, 135, 180, 225, 270, 315),
    ...     grouping="Gender"
    ... )
    """
    # Select circumplex scales and grouping variable (if applicable)
    if grouping is not None:
        bs_input = data[scales + [grouping]].copy()
        bs_input["Group"] = bs_input[grouping].astype("category")

        # Check if more than one contrast is possible
        if contrast != "none" and len(bs_input["Group"].cat.categories) != 2:
            raise ValueError(
                "Only two groups can be contrasted at a time. Set contrast = 'none' or use a dichotomous grouping variable."
            )
    else:
        bs_input = data[scales].copy()
        bs_input["Group"] = "All"

    # Perform listwise deletion if requested
    if listwise:
        bs_input = bs_input.dropna()

    # Calculate mean observed scores
    scores = bs_input.groupby("Group", observed=False)[scales].mean().reset_index()
    scores = scores.rename_axis("label").reset_index()

    # Define bootstrap function
    def bs_function(data, index, angles, contrast, listwise):
        resample = data.iloc[index]
        scores_r = resample.groupby("Group", observed=False)[scales].mean()
        return ssm_by_group(scores_r, angles, contrast)

    # Perform bootstrapping
    bs_output = ssm_bootstrap(
        bs_input=bs_input,
        bs_function=bs_function,
        angles=angles,
        boots=boots,
        interval=interval,
        contrast=contrast,
        listwise=listwise,
        strata=bs_input["Group"],
    )

    # Select and label results
    group_levels = bs_input["Group"].unique()
    if contrast == "none":
        row_data = bs_output
        row_labels = group_levels
    else:
        row_data = bs_output.iloc[-1:]
        row_labels = [f"{group_levels[1]} - {group_levels[0]}"]

    results = row_data.copy()
    results["label"] = row_labels

    # Collect analysis details
    details = {
        "boots": boots,
        "interval": interval,
        "listwise": listwise,
        "angles": angles,
        "contrast": contrast,
        "score_type": "Mean",
        "results_type": "Profile" if contrast == "none" else "Contrast",
    }

    call_str = (
        f"ssm_analyze_means(data, scales={scales}, angles={angles}, "
        f"grouping={grouping}, contrast={contrast}, boots={boots}, "
        f"interval={interval}, listwise={listwise})"
    )

    return {"results": results, "details": details, "call": call_str, "scores": scores}


def ssm_by_group(
    scores: pd.DataFrame, angles: Tuple[float], contrast: str
) -> np.ndarray:
    """
    Calculate SSM parameters for each group, potentially with contrast.

    Args:
        scores (pd.DataFrame): DataFrame containing scores for each group.
        angles (List[float]): Angular displacement of each circumplex scale (in radians).
        contrast (str): Type of contrast to run ("none", "model", or "test").

    Returns:
        np.ndarray: Array of SSM parameters for each group (and contrast if applicable).
    """
    # Convert scores DataFrame to numpy array
    scores_array = scores.to_numpy()

    # To model contrast, subtract scores then SSM
    if contrast == "model":
        contrast_scores = scores_array[1] - scores_array[0]
        scores_array = np.vstack([scores_array, contrast_scores])

    # Calculate parameters per group
    results = group_parameters(scores_array, angles)

    # To test contrast, SSM then subtract parameters
    if contrast == "test":
        contrast_params = results[6:] - results[:6]
        results = np.concatenate([results, contrast_params])

    return results


def group_parameters(scores: np.ndarray, angles: Tuple[float]) -> np.ndarray:
    """
    Calculate the SSM parameters as a vector for each group where rows are groups.

    Args:
        scores (np.ndarray): 2D array where each row represents a group's scores.
        angles (List[float]): Angular displacement of each circumplex scale (in radians).

    Returns:
        np.ndarray: 1D array of SSM parameters for all groups.
    """
    n = scores.shape[0]  # Number of groups
    out = np.zeros(n * 6)  # Initialize output array

    for i in range(n):
        out[i * 6 : (i + 1) * 6] = ssm_parameters(scores[i], angles)

    return out


def ssm_parameters(
    scores: NDArray, angles: Tuple[float, ...], bounds=BOUNDS
) -> NDArray:
    """Calculate SSM parameters (without confidence intervals) for a set of scores.

    Args:
        scores (np.array): A numeric vector (or single row dataframe) containing one score for each of a
            set of circumplex scales.
        angles (tuple): A numeric vector containing the angular displacement of each circumplex scale
            included in `scores`.
        bounds (tuple, optional): The bounds for each of the parameters of the curve optimisation.
            Defaults to ([0, 0, -1], [np.inf, 360, 1]).

    Returns:
        tuple: A tuple containing the elevation, x-value, y-value, amplitude, displacement, and R2 fit of the SSM curve.

    Examples:
        # >>> scores = np.array([-0.5, 0, 0.25, 0.51, 0.52, 0.05, -0.26, -0.7])
        # >>> angles = OCTANTS
        # >>> results = ssm_parameters_cpp(scores, angles)
        # >>> [round(i, 3) for i in results]
        [-0.016, -0.478, 0.333, 0.582, 145.158, 0.967]
    """

    # noinspection PyTupleAssignmentBalance
    # NOTE: Bug - Sometimes returns displacement at the trough, not the crest, so 180 degrees off
    # This was addressed by setting the lower bound of amplitude to 0, not -np.inf. Need a less hard-coded solution
    param, covariance = curve_fit(
        utils.cosine_form, xdata=np.deg2rad(angles), ydata=scores, bounds=bounds
    )
    r2 = utils.r2_score(scores, utils.cosine_form(np.deg2rad(angles), *param))
    ampl, disp, elev = param

    def polar2cart(r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    xval, yval = polar2cart(ampl, disp)
    return np.array([elev, xval, yval, ampl, np.rad2deg(disp), r2])


def ssm_bootstrap(
    bs_input: pd.DataFrame,
    bs_function: Callable,
    angles: Tuple[float, ...],
    boots: int,
    interval: float,
    contrast: str,
    listwise: bool,
    strata: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Perform bootstrap to get confidence intervals around SSM parameters.

    Args:
        bs_input (pd.DataFrame): Input data for bootstrapping.
        bs_function (Callable): Function to calculate SSM parameters.
        angles (List[float]): Angular displacement of each circumplex scale (in radians).
        boots (int): Number of bootstrap resamples.
        interval (float): Confidence interval level.
        contrast (str): Type of contrast to run ("none", "model", or "test").
        listwise (bool): Whether to use listwise deletion for missing values.
        strata (Optional[pd.Series]): Series to use for stratified sampling.

    Returns:
        pd.DataFrame: DataFrame containing bootstrap results.
    """
    # Perform bootstrapping
    bootstrap_results = []
    for _ in range(boots):
        if strata is not None:
            # Note: Using observed=False for categorical data
            # The DeprecationWarning about operating on grouping columns is expected
            # but won't affect functionality
            resample = (
                bs_input.groupby(strata, observed=False)
                .apply(lambda x: x.sample(n=len(x), replace=True))
                .reset_index(drop=True)
            )
        else:
            resample = bs_input.sample(n=len(bs_input), replace=True)
        bootstrap_results.append(
            bs_function(resample, range(len(resample)), angles, contrast, listwise)
        )

    bs_t = np.array(bootstrap_results)

    # Calculate point estimates
    bs_est = pd.DataFrame(
        bs_function(bs_input, range(len(bs_input)), angles, contrast, listwise).reshape(
            -1, 6
        ),
        columns=[f"{p}_est" for p in ["e", "x", "y", "a", "d", "fit"]],
    )

    # Calculate confidence intervals
    bs_lci = pd.DataFrame(
        np.percentile(bs_t, (1 - interval) / 2 * 100, axis=0).reshape(-1, 6),
        columns=[f"{p}_lci" for p in ["e", "x", "y", "a", "d", "fit"]],
    )
    bs_uci = pd.DataFrame(
        np.percentile(bs_t, (1 + interval) / 2 * 100, axis=0).reshape(-1, 6),
        columns=[f"{p}_uci" for p in ["e", "x", "y", "a", "d", "fit"]],
    )

    # Combine results and convert radians to degrees for displacement
    results = pd.concat([bs_est, bs_lci, bs_uci], axis=1)
    results["d_est"] = results["d_est"] % 360  # normalize to 0-360

    return results


def ssm_analyze_corrs(
    data: pd.DataFrame,
    scales: List[str],
    angles: Tuple[float, ...],
    measures: List[str],
    grouping: Optional[str] = None,
    contrast: str = "none",
    boots: int = 2000,
    interval: float = 0.95,
    listwise: bool = True,
    measures_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Perform analyses using the correlation-based Structural Summary Method.

    This function calculates SSM parameters based on the correlations between
    circumplex scales and external measures, optionally grouped by a categorical variable.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing at least circumplex scales and measures.
    scales : List[str]
        The variable names for the circumplex scales to be analyzed.
    angles : Tuple[float]
        Angular displacement of each circumplex scale (in degrees).
    measures : List[str]
        Variables to be correlated with the circumplex scales.
    grouping : Optional[str], optional
        Variable name that indicates group membership of each observation.
    contrast : str, optional
        Type of contrast to run ("none", "model", or "test").
        Default is "none".
    boots : int, optional
        Number of bootstrap resamples for estimating confidence intervals.
        Default is 2000.
    interval : float, optional
        Confidence level for estimating the confidence intervals.
        Default is 0.95.
    listwise : bool, optional
        Whether to use listwise deletion for missing values.
        Default is True.
    measures_labels : Optional[List[str]], optional
        Labels for each measure provided in measures.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the results and description of the analysis:
        - results: DataFrame with SSM parameters for each measure-group combination
        - details: Dictionary with analysis details
        - call: String representation of the function call
        - scores: DataFrame with correlation scores between measures and scales

    Examples
    --------
    >>> results_dict = ssm_analyze_corrs(
    ...     data,
    ...     scales=["PA", "BC", "DE", "FG", "HI", "JK", "LM", "NO"],
    ...     angles=(0, 45, 90, 135, 180, 225, 270, 315),
    ...     measures=["Extraversion", "Neuroticism"],
    ...     measures_labels=["Extraversion Scale", "Neuroticism Scale"]
    ... )
    """
    # Select circumplex scales, measure variables, and grouping variable
    if grouping is not None:
        bs_input = data[scales + measures + [grouping]].copy()
        # Perform listwise deletion if requested
        if listwise:
            bs_input = bs_input.dropna()
        bs_input["Group"] = bs_input[grouping].astype("category")
        if bs_input["Group"].nunique() != data[grouping].nunique():
            warnings.warn("Listwise deletion removed some groups.")

    else:
        bs_input = data[scales + measures].copy()
        # Perform listwise deletion if requested
        if listwise:
            bs_input = bs_input.dropna()
        bs_input["Group"] = "All"
        bs_input["Group"] = bs_input["Group"].astype("category")
    if bs_input.empty:
        raise ValueError("No data remains after listwise deletion.")

    # Check that this combination of arguments is executable
    n_measures = len(measures)
    n_groups = bs_input["Group"].nunique()
    if contrast != "none":
        contrast_measures = n_measures == 2 and n_groups == 1
        contrast_groups = n_measures == 1 and n_groups == 2
        if not (contrast_measures or contrast_groups):
            raise ValueError(
                "No valid contrasts were possible. To contrast measures, ensure "
                "there are 2 measures and no grouping variable. To contrast groups, "
                "ensure there is 1 measure and a dichotomous grouping variable."
            )

    # Select and label results
    if measures_labels is None:
        measure_names = measures
    else:
        measure_names = measures_labels

    # Calculate observed scores (i.e., correlations)
    cs = bs_input[scales].values
    mv = bs_input[measures].values
    grp = bs_input["Group"].astype("category").cat.codes.values
    scores = corr_scores(cs, mv, grp, listwise, scales)
    scores_df = pd.DataFrame(scores, columns=scales)
    scores_df["Group"] = np.repeat(bs_input["Group"].unique(), len(measures))
    scores_df["Measure"] = np.tile(measure_names, n_groups)
    if grouping is not None:
        scores_df["label"] = scores_df["Group"].astype(str) + "_" + scores_df["Measure"]
    else:
        scores_df["label"] = scores_df["Measure"]

    # Define bootstrap function
    def bs_function(data, index, angles, contrast, listwise):
        resample = data.iloc[index]
        grp = resample["Group"].astype("category").cat.codes.values
        cs = resample[scales].values
        mv = resample[measures].values
        scores_r = corr_scores(cs, mv, grp, listwise)
        scores_r = scores_r.drop(columns=["Group", "Measure"])
        return ssm_by_group(scores_r, angles, contrast)

    # Perform bootstrapping
    bs_output = ssm_bootstrap(
        bs_input=bs_input,
        bs_function=bs_function,
        angles=angles,
        boots=boots,
        interval=interval,
        contrast=contrast,
        listwise=listwise,
        strata=bs_input["Group"],
    )

    # Select and label results
    group_names = bs_input["Group"].cat.categories
    if contrast == "none":
        row_data = bs_output
        grp_labels = np.repeat(group_names, len(measures))
        msr_labels = np.tile(measure_names, n_groups)
        if grouping is not None:
            lbl_labels = grp_labels + "_" + msr_labels
        else:
            lbl_labels = msr_labels
        results = row_data.assign(
            label=lbl_labels, Group=grp_labels, Measure=msr_labels
        )
    else:
        row_data = bs_output.iloc[-1:].copy()
        if n_measures == 2:
            row_labels = [f"{measure_names[1]} - {measure_names[0]}"]
        else:
            row_labels = [f"{measure_names[0]}: {group_names[1]} - {group_names[0]}"]
        results = row_data.assign(label=row_labels)

    # Collect analysis details
    details = {
        "boots": boots,
        "interval": interval,
        "listwise": listwise,
        "angles": angles,
        "contrast": contrast,
        "score_type": "Correlation",
        "results_type": "Profile" if contrast == "none" else "Contrast",
    }

    call_str = (
        f"ssm_analyze_corrs(data, scales={scales}, angles={angles}, "
        f"measures={measures}, grouping={grouping}, contrast={contrast}, "
        f"boots={boots}, interval={interval}, listwise={listwise}, "
        f"measures_labels={measures_labels})"
    )

    return {
        "results": results,
        "details": details,
        "call": call_str,
        "scores": scores_df,
    }


def corr_scores(
    scores: Union[np.ndarray, pd.DataFrame],
    measures: Union[np.ndarray, pd.DataFrame],
    grouping: Union[np.ndarray, pd.Series],
    listwise: bool,
    scales: List[str] | None = None,
) -> pd.DataFrame:
    """
    Calculate the correlation of each measure with each scale by group.

    Args:
        scores (Union[np.ndarray, pd.DataFrame]): Circumplex scale scores.
        measures (Union[np.ndarray, pd.DataFrame]): Measure variable scores.
        grouping (Union[np.ndarray, pd.Series]): Group codes.
        listwise (bool): Whether to use listwise deletion (True) or pairwise deletion (False).
        scales (List[str], optional): Names of the circumplex scales. If None, will use column names if cs is a DataFrame, else will use default names.

    Returns:
        pd.DataFrame: Correlation scores.
    """
    # Convert inputs to numpy arrays if they're not already
    cs_array = scores.values if isinstance(scores, pd.DataFrame) else scores
    mv_array = measures.values if isinstance(measures, pd.DataFrame) else measures
    grp_array = (
        grouping.values if isinstance(grouping, (pd.Series, pd.DataFrame)) else grouping
    )

    levels = np.unique(grp_array)
    ng = len(levels)
    pm, ps = mv_array.shape[1], cs_array.shape[1]
    out = np.zeros((ng * pm, ps))

    if ng == 1:
        if listwise:
            # Single group and LWD
            out = np.corrcoef(mv_array.T, cs_array.T)[:pm, pm:]
        else:
            # Single group and PWD
            for m in range(pm):
                x = mv_array[:, m]
                for s in range(ps):
                    y = cs_array[:, s]
                    out[m, s] = pairwise_r(x, y)
    else:
        if listwise:
            # Multiple groups and LWD
            for g, level in enumerate(levels):
                mask = grp_array == level
                gcs = cs_array[mask]
                gmv = mv_array[mask]
                out[g * pm : (g + 1) * pm, :] = np.corrcoef(gmv.T, gcs.T)[:pm, pm:]
        else:
            # Multiple groups and PWD
            for g, level in enumerate(levels):
                mask = grp_array == level
                gcs = cs_array[mask]
                gmv = mv_array[mask]
                for m in range(pm):
                    x = gmv[:, m]
                    for s in range(ps):
                        y = gcs[:, s]
                        out[g * pm + m, s] = pairwise_r(x, y)

    # Create a DataFrame from the output
    if scales is None:
        if isinstance(scores, pd.DataFrame):
            scales = scores.columns.tolist()
        else:
            scales = [f"Scale_{i + 1}" for i in range(ps)]

    df_out = pd.DataFrame(out, columns=scales)

    # Add group and measure information
    if isinstance(measures, pd.DataFrame):
        measure_names = measures.columns.tolist()
    else:
        measure_names = [f"Measure_{i + 1}" for i in range(pm)]

    df_out["Group"] = np.repeat(levels, pm)
    df_out["Measure"] = np.tile(measure_names, ng)

    return df_out


def pairwise_r(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Pearson correlation coefficient using pairwise deletion.

    Args:
        x (np.ndarray): First array of values.
        y (np.ndarray): Second array of values.

    Returns:
        float: Pearson correlation coefficient.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    return np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan


if __name__ == "__main__":
    ######## SCRATCH ########
    from importlib.resources import files

    import matplotlib.pyplot as plt

    _jz2017_path = str(files("circumplex.data").joinpath("jz2017.csv"))
    data = pd.read_csv(_jz2017_path)

    results = ssm_analyze(
        data=data,
        scales=["PA", "BC", "DE", "FG", "HI", "JK", "LM", "NO"],
        angles=(90, 135, 180, 225, 270, 315, 0, 45),
        # grouping="Gender",
        # contrast='model',
        measures=["NARPD", "ASPD"],
        # measures_labels=['Narcissistic PD', 'Antisocial PD'],
    )
    print(results.scores)
    # print(results)
    print(results.summary())

    # fig = ssm_plot(results)
    fig, ax = results.profile_plot(incl_elev=True)
    plt.show()
