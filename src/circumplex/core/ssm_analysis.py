import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from typing import Optional, List, Union, Callable
from circumplex.core.ssm_results import SSMResults
from circumplex.core.utils import OCTANTS, cosine_form, r2_score

BOUNDS = ([0, 0, -np.inf], [np.inf, 2 * np.pi, np.inf])


def ssm_analyze(
    data: pd.DataFrame,
    scales: List[str],
    angles: Optional[List[float]] = OCTANTS,
    measures: Optional[List[str]] = None,
    grouping: Optional[str] = None,
    contrast: Optional[str] = "none",
    boots: int = 2000,
    interval: float = 0.95,
    listwise: bool = True,
    measures_labels: Optional[List[str]] = None,
) -> SSMResults:
    """
    Perform analyses using the Structural Summary Method.

    This function calculates SSM parameters with bootstrapped confidence intervals for various
    analysis types. Depending on the arguments supplied, it performs either mean-based or
    correlation-based analyses, uses one or more groups to stratify the data, and calculates
    contrasts between groups or measures.

    Args:
        data (pd.DataFrame): A DataFrame containing at least circumplex scales.
        scales (List[str]): The variable names for the circumplex scales to be analyzed.
        angles (Optional[List[float]]): Angular displacement of each circumplex scale (in degrees).
        measures (Optional[List[str]]): Variables to be correlated with the circumplex scales.
        grouping (Optional[str]): Variable name that indicates group membership of each observation.
        contrast (str): Type of contrast to run ("none", "model", or "test").
        boots (int): Number of bootstrap resamples for estimating confidence intervals.
        interval (float): Confidence level for estimating the confidence intervals.
        listwise (bool): Whether to use listwise deletion for missing values.
        measures_labels (Optional[List[str]]): Labels for each measure provided in measures.

    Returns:
        SSMResults: An object containing the results and description of the analysis.
    """

    assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"
    assert all(
        scale in data.columns for scale in scales
    ), "All scales must be present in data"
    assert isinstance(angles, list) and all(
        isinstance(a, (int, float)) for a in angles
    ), "angles must be a list of numbers"
    assert isinstance(boots, int) and boots > 0, "boots must be a positive integer"
    assert 0 < interval < 1, "interval must be between 0 and 1"
    assert isinstance(listwise, bool), "listwise must be a boolean"
    assert contrast in [
        "none",
        "model",
        "test",
    ], "contrast must be 'none', 'model', or 'test'"

    if measures is not None:
        assert all(
            measure in data.columns for measure in measures
        ), "All measures must be present in data"

    if grouping is not None:
        assert grouping in data.columns, "grouping must be a column in data"

    if measures_labels is not None:
        assert len(measures_labels) == len(
            measures
        ), "measures_labels must have the same length as measures"

    # Convert angles from degrees to radians
    angles_rad = np.deg2rad(angles)

    # Determine analysis type and forward to appropriate subfunction
    if measures is not None:
        if grouping is not None:
            # Multiple group correlations
            results = ssm_analyze_corrs(
                data,
                scales,
                angles_rad,
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
                angles_rad,
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
                data, scales, angles_rad, grouping, contrast, boots, interval, listwise
            )
        else:
            # Single group means
            if contrast != "none":
                raise ValueError(
                    "Without specifying measures or grouping, no contrasts are possible. "
                    "Set contrast = 'none' or add the measures or grouping arguments."
                )
            results = ssm_analyze_means(
                data, scales, angles_rad, None, "none", boots, interval, listwise
            )

    # Calculate scores
    if measures is not None:
        scores = corr_scores(
            data[scales], data[measures], data[grouping] if grouping else None, listwise
        )
    else:
        scores = (
            data[scales]
            .groupby(data[grouping] if grouping else pd.Series(["All"] * len(data)))
            .mean()
        )

    # Create the call string
    call_str = (
        f"ssm_analyze(data, scales={scales}, angles={angles}, measures={measures}, "
        f"grouping={grouping}, contrast={contrast}, boots={boots}, "
        f"interval={interval}, listwise={listwise}, measures_labels={measures_labels})"
    )

    # Create and return the SSMResults object
    return SSMResults(
        results=results["results"],
        scores=results["scores"],
        details=results["details"],
        call=call_str,
    )


def ssm_analyze_means(
    data: pd.DataFrame,
    scales: List[str],
    angles: List[float],
    grouping: Optional[str] = None,
    contrast: str = "none",
    boots: int = 2000,
    interval: float = 0.95,
    listwise: bool = True,
) -> dict:
    """
    Perform analyses using the mean-based Structural Summary Method.

    Args:
        data (pd.DataFrame): A DataFrame containing at least circumplex scales.
        scales (List[str]): The variable names for the circumplex scales to be analyzed.
        angles (List[float]): Angular displacement of each circumplex scale (in radians).
        grouping (Optional[str]): Variable name that indicates group membership of each observation.
        contrast (str): Type of contrast to run ("none", "model", or "test").
        boots (int): Number of bootstrap resamples for estimating confidence intervals.
        interval (float): Confidence level for estimating the confidence intervals.
        listwise (bool): Whether to use listwise deletion for missing values.

    Returns:
        dict: A dictionary containing the results and description of the analysis.
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
        "angles": np.rad2deg(angles),
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
    scores: pd.DataFrame, angles: List[float], contrast: str
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


def group_parameters(scores: np.ndarray, angles: List[float]) -> np.ndarray:
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


def ssm_parameters(scores: np.ndarray, angles: List[float], bounds=BOUNDS) -> np.array:
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
        cosine_form, xdata=angles, ydata=scores, bounds=bounds
    )
    r2 = r2_score(scores, cosine_form(angles, *param))
    ampl, disp, elev = param

    def polar2cart(r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    xval, yval = polar2cart(ampl, disp)
    return np.array([elev, xval, yval, ampl, disp, r2])


def ssm_bootstrap(
    bs_input: pd.DataFrame,
    bs_function: Callable,
    angles: List[float],
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
    for col in ["d_est", "d_lci", "d_uci"]:
        results[col] = np.rad2deg(results[col])

    return results


def ssm_analyze_corrs(
    data: pd.DataFrame,
    scales: List[str],
    angles: List[float],
    measures: List[str],
    grouping: Optional[str] = None,
    contrast: str = "none",
    boots: int = 2000,
    interval: float = 0.95,
    listwise: bool = True,
    measures_labels: Optional[List[str]] = None,
) -> dict:
    """
    Perform analyses using the correlation-based Structural Summary Method.

    Args:
        data (pd.DataFrame): A DataFrame containing at least circumplex scales and measures.
        scales (List[str]): The variable names for the circumplex scales to be analyzed.
        angles (List[float]): Angular displacement of each circumplex scale (in radians).
        measures (List[str]): Variables to be correlated with the circumplex scales.
        grouping (Optional[str]): Variable name that indicates group membership of each observation.
        contrast (str): Type of contrast to run ("none", "model", or "test").
        boots (int): Number of bootstrap resamples for estimating confidence intervals.
        interval (float): Confidence level for estimating the confidence intervals.
        listwise (bool): Whether to use listwise deletion for missing values.
        measures_labels (Optional[List[str]]): Labels for each measure provided in measures.

    Returns:
        dict: A dictionary containing the results and description of the analysis.
    """
    # Select circumplex scales, measure variables, and grouping variable
    if grouping is not None:
        bs_input = data[scales + measures + [grouping]].copy()
        bs_input["Group"] = bs_input[grouping].astype("category")
    else:
        bs_input = data[scales + measures].copy()
        bs_input["Group"] = "All"
        bs_input["Group"] = bs_input["Group"].astype("category")

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

    # Perform listwise deletion if requested
    if listwise:
        bs_input = bs_input.dropna()

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
        "angles": np.rad2deg(angles),
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
    scales: List[str] = None,
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
    from visualization import ssm_plot, ssm_plot_profile

    _jz2017_path = str(files("circumplex.data").joinpath("jz2017.csv"))
    data = pd.read_csv(_jz2017_path)

    results = ssm_analyze(
        data=data,
        scales=["PA", "BC", "DE", "FG", "HI", "JK", "LM", "NO"],
        angles=[90, 135, 180, 225, 270, 315, 0, 45],
        grouping="Gender",
        # contrast='model',
        measures=["NARPD", "ASPD"],
        # measures_labels=['Narcissistic PD', 'Antisocial PD'],
    )
    print(results.scores)
    # print(results)
    print(results.summary())

    # fig = ssm_plot(results)
    fig, ax = results.profile_plot()
    plt.show()
