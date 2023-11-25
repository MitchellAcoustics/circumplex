# %%

import matplotlib.pyplot as plt
from matplotlib import colormaps
from cycler import cycler
import numpy as np
import pandas as pd
import math
from scipy.optimize import curve_fit

OCTANTS = (0, 45, 90, 135, 180, 225, 270, 315)
BOUNDS = ([0, 0, -np.inf], [np.inf, 360, np.inf])


class SSMParams(object):
    """
    A class to store the results of a single SSM analysis.

    Attributes:
        scores (np.array): A numeric vector (or single row dataframe) containing one score for each of a
            set of circumplex scales.
        angles (tuple): A numeric vector containing the angular displacement of each circumplex scale
            included in `scores`.
        scales (list): A list of the names of the circumplex scales included in `scores`.
        group (list): A list of the names of the groups included in `scores`.
        measure (list): A list of the names of the measures included in `scores`.
        elevation (float): The elevation of the SSM curve.
        xval (float): The x-value of the SSM curve.
        yval (float): The y-value of the SSM curve.
        amplitude (float): The amplitude of the SSM curve.
        displacement (float): The displacement of the SSM curve.
        r2 (float): The R2 fit of the SSM curve.
    """

    def __init__(
        self,
        scores: np.array,
        scales: list,
        angles: tuple = OCTANTS,
        group: list = None,
        measure: list | str = None,
        bounds: tuple = BOUNDS,
    ):
        self.scores = scores
        self.angles = angles
        self.scales = scales
        self.group = group
        self.measure = measure
        (
            self.elevation,
            self.xval,
            self.yval,
            self.amplitude,
            self.displacement,
            self.r2,
        ) = ssm_parameters(self.scores, self.angles, bounds=bounds)

    @property
    def label(self) -> str:
        """Return a label for the SSMParams object."""
        if self.group is not None and self.measure is not None:
            return f"{self.group}_{self.measure}"
        elif self.measure is not None:
            return self.measure
        elif self.group is not None:
            return self.group
        else:
            return "SSM"

    @property
    def table(self) -> pd.DataFrame:
        """Return a table of the results."""
        scale_angle = {scale: angle for scale, angle in zip(self.scales, self.angles)}
        return pd.DataFrame(
            self.params | scale_angle,
            index=[self.label],
        )

    @property
    def params(self) -> dict:
        return {
            "label": self.label,
            "group": self.group,
            "measure": self.measure,
            "elevation": self.elevation,
            "xval": self.xval,
            "yval": self.yval,
            "amplitude": self.amplitude,
            "displacement": self.displacement,
            "r2": self.r2,
        }

    def __repr__(self):
        # TODO: Add param results
        return f"SSMParams({self.label}, scores={self.scores}, angles={self.angles})"

    def __str__(self):
        # TODO: Add param results
        return (
            "====================================\n"
            f"Measure:          {self.measure}\n"
            f"Group:            {self.group}\n"
            f"Scales:           {self.scales}\n"
            f"Scale Angles:     {self.angles}\n"
            f"\nProfile [All]:\n"
            f"                  Estimate\n"
            f"Elevation:        {round(self.elevation, 3)}\n"
            f"X-Value:          {round(self.xval, 3)}\n"
            f"Y-Value:          {round(self.yval, 3)}\n"
            f"Amplitude:        {round(self.amplitude, 3)}\n"
            f"Displacement:     {round(self.displacement, 3)}\n"
            f"R2:               {round(self.r2, 3)}\n"
        )

    def profile_plot(self, ax=None, **kwargs) -> plt.Axes:
        """
        Plot the SSM profile.

        Returns:
            plt.Axes: A tuple containing the figure and axis objects.
        """
        return profile_plot(
            self.amplitude,
            self.displacement,
            self.elevation,
            self.r2,
            self.angles,
            self.scores,
            self.label,
            ax=ax,
            **kwargs,
        )

    def plot(self) -> plt.Axes:
        """
        Plot the results in the circumplex

        Returns:
            plt.Axes: A tuple containing the figure and axis objects.
        """
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.plot(
            np.deg2rad(self.displacement),
            self.amplitude,
            color="black",
            marker="o",
            markersize=10,
        )


class SSMResults(object):
    """
    A class to store the results of a SSM analysis.

    Attributes:
        results (list[SSMParams]): A list containing the SSMParams results of the SSM analysis.
        measures (list): A list of the names of the measures included in `scores`.
        grouping (list): A list of the names of the groups included in `scores`.
    """

    def __init__(
        self, results: list[SSMParams] | SSMParams, measures=None, grouping=None
    ):
        if isinstance(results, SSMParams):
            measures = [results.measure]
            grouping = [results.group]
            results = [results]

        self.results = results
        self.measures = measures
        self.grouping = grouping

    def __str__(self):
        return "\n".join([str(res) for res in self.results])

    @property
    def groups(self) -> set:
        """Return a list of the groups included in the analysis."""
        return set(val.group for val in self.results if val.group is not None)

    @property
    def labels(self) -> set:
        """Return a list of the labels included in the analysis."""
        return set(val.label for val in self.results)

    @property
    def table(self) -> pd.DataFrame:
        """
        Return a table of the results.

        Returns:
            pd.DataFrame: A dataframe containing the results of the SSM analysis.
        """
        df = pd.DataFrame()
        for val in self.results:
            df = pd.concat([df, val.table])
        return df

    def query_label(self, label: str) -> SSMParams:
        """Query the results for a specific SSMParams by label."""
        for res in self.results:
            if res.label == label:
                return res
        raise ValueError(
            f"No results found for {label}"
        )  # Raised if the label is never found

    def plot(self, colors=None, legend=True, *args, **kwargs) -> plt.Axes:
        """
        Plot the results in the circumplex

        Returns:
            plt.Axes: A tuple containing the figure and axis objects.
        """
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        if colors is None:
            colors = colormaps.get_cmap("tab20").colors
        colors = iter(colors)
        for res in self.results:
            ax.plot(
                np.deg2rad(res.displacement),
                res.amplitude,
                color=next(colors),
                marker="o",
                markersize=10,
                label=res.label,
            )
        fig.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
        return fig, ax

    def profile_plots(self, axes=None) -> plt.Axes:
        """
        Plot the SSM profiles.

        Returns:
            plt.Axes: A tuple containing the figure and axis objects.
        """
        if axes is None:
            fig, axes = plt.subplots(
                nrows=len(self.results),
                figsize=(8, 4 * len(self.results)),
            )
        for i, res in enumerate(self.results):
            # BUG: Raises error if we only need a single plot
            fig, ax = res.profile_plot(ax=axes.flatten()[i])
        plt.tight_layout()
        # plt.show()
        return fig, axes


# %%


def ssm_analyse(
    data: pd.DataFrame,
    scales: list,
    measures: list | None = None,
    grouping: list | None = None,
    angles: tuple = OCTANTS,
    grouped_angles: dict = None,
) -> SSMResults:
    """
    Analyse a set of data using the SSM method.

    Args:
        data (pd.DataFrame): A dataframe containing the data to be analysed.
        scales (list): A list of the names of the circumplex scales to be included in the analysis.
        measures (list, optional): A list of the names of the measures to be included in the analysis. Defaults to None.
        grouping (list, optional): A list of the names of the groups to be included in the analysis. Defaults to None.
        angles (tuple, optional): A tuple containing the angular displacement of each circumplex scale
            included in `scores`. Defaults to OCTANTS.
        grouped_angles (dict, optional): A dictionary containing the angular displacement of each circumplex scale
            included in `scores` for each group. Defaults to None.

    Returns:
        SSMResults: A SSMResults object containing the results of the analysis.

    """
    if grouping is not None and measures is not None:
        return ssm_analyse_grouped_corrs(
            data, scales, measures, grouping, angles, grouped_angles
        )
    elif measures is not None:
        return ssm_analyse_corrs(data, scales, measures, angles)
    elif grouping is not None:
        return ssm_analyse_means(data, scales, grouping, angles, grouped_angles)
    else:
        ssm = SSMParams(data[scales].mean(), scales, angles)
        # ssm.param_calc()
        return SSMResults(ssm)


def ssm_analyse_grouped_corrs(
    data: pd.DataFrame,
    scales: tuple,
    measures: list,
    grouping: list,
    angles: tuple = OCTANTS,
    grouped_angles: dict = None,
) -> SSMResults:
    """
    Perform SSM analysis of correlations for a set of grouped data.

    Args:
        data (pd.DataFrame): A dataframe containing the data to be analysed.
        scales (tuple): A list of the names of the circumplex scales to be included in the analysis.
        measures (list): A list of the names of the measures to be included in the analysis.
        grouping (list): A list of the names of the groups to be included in the analysis.
        angles (tuple, optional): A tuple containing the angular displacement of each circumplex scale
            included in `scores`. Defaults to OCTANTS.

    Returns:
            SSMResults: A SSMResults object containing the results of the analysis.
    """
    res = []
    for group_var in grouping:
        for group, group_data in data.groupby(group_var):
            if grouped_angles is not None:
                angles = grouped_angles[group]  # grouped angles will override angles
            try:
                res.append(
                    ssm_analyse_corrs(
                        group_data, scales, measures, angles, group=group
                    ).results[0]
                )
            except ValueError as e:
                print(f"Error: {e} | in {group_var} = {group}")

    return SSMResults(res, measures, grouping)


def ssm_analyse_corrs(
    data: pd.DataFrame,
    scales: tuple,
    measures: list,
    angles: tuple = OCTANTS,
    group: str | None = None,
) -> SSMResults:
    """
    Perform SSM analysis of correlations for a set of data.

    Args:
        data (pd.DataFrame): A dataframe containing the data to be analysed.
        scales (tuple): A list of the names of the circumplex scales to be included in the analysis.
        measures (list): A list of the names of the measures to be included in the analysis.
        angles (tuple, optional): A tuple containing the angular displacement of each circumplex scale
            included in `scores`. Defaults to OCTANTS.
        group (str, optional): The name of the group to be included in the analysis. Defaults to None.

    Returns:
        SSMResults: A SSMResults object containing the results of the analysis.
    """
    res = []
    for measure in measures:
        r = data[scales].corrwith(data[measure])
        ssm = SSMParams(r, scales, angles, measure=measure, group=group)
        # ssm.param_calc()
        res.append(ssm)

    return SSMResults(res, measures)


def ssm_analyse_means(
    data: pd.DataFrame,
    scales: tuple,
    grouping: list,
    angles: tuple = OCTANTS,
    grouped_angles: dict = None,
) -> SSMResults:
    """
    Perform SSM analysis of means for a set of data.

    Args:
        data (pd.DataFrame): A dataframe containing the data to be analysed.
        scales (tuple): A list of the names of the circumplex scales to be included in the analysis.
        grouping (list): A list of the names of the groups to be included in the analysis.
        angles (tuple, optional): A tuple containing the angular displacement of each circumplex scale
            included in `scores`. Defaults to OCTANTS.

    Returns:
        SSMResults: A SSMResults object containing the results of the analysis.
    """
    means = data.groupby(grouping)[scales].mean()
    res = []
    for group, scores in means.iterrows():
        if grouped_angles is not None:
            angles = grouped_angles[group]
        scores = means.loc[group]
        ssm = SSMParams(scores, scales, angles, group=group)
        # ssm.param_calc()
        res.append(ssm)

    return SSMResults(res, grouping=grouping)


def cosine_form(theta, ampl, disp, elev):
    """Cosine function with amplitude, dispersion and elevation parameters."""
    return elev + ampl * np.cos(np.deg2rad(theta - disp))


def _r2_score(y_true: np.array, y_pred: np.array):
    """Calculate the R2 score for a set of predictions."""
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - (ss_res / ss_tot)


def ssm_parameters(scores, angles, bounds=BOUNDS) -> tuple:
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
        >>> scores = np.array([-0.5, 0, 0.25, 0.51, 0.52, 0.05, -0.26, -0.7])
        >>> angles = OCTANTS
        >>> results = ssm_parameters(scores, angles)
        >>> [round(i, 3) for i in results]
        [-0.016, -0.478, 0.333, 0.582, 145.158, 0.967]
    """

    # noinspection PyTupleAssignmentBalance
    # NOTE: Bug - Sometimes returns displacement at the trough, not the crest, so 180 degrees off
    # This was addressed by setting the lower bound of amplitude to 0, not -np.inf. Need a less hard-coded solution
    param, covariance = curve_fit(
        cosine_form, xdata=angles, ydata=scores, bounds=bounds
    )
    r2 = _r2_score(scores, cosine_form(angles, *param))
    ampl, disp, elev = param

    def polar2cart(r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    xval, yval = polar2cart(ampl, np.deg2rad(disp))
    return elev, xval, yval, ampl, disp, r2


def profile_plot(
    amplitude,
    displacement,
    elevation,
    r2,
    angles,
    scores,
    label,
    ax=None,
    reorder_scales=True,
    incl_pred=True,
    incl_fit=True,
    incl_disp=True,
    incl_amp=True,
) -> plt.Axes:
    """
    Plot the SSM profile.

    Args:
        reorder_scales:
        incl_pred:
        incl_fit:
        incl_disp:
        incl_amp:

    Returns:
        plt.Axes: A tuple containing the figure and axis objects.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    if reorder_scales:
        angles, scores = sort_angles(angles, scores)
        if angles[-1] == 360:
            angles = (0,) + angles
            scores = (scores[-1],) + scores

    if incl_pred:
        thetas = np.linspace(0, 360, 1000)
        fit = cosine_form(thetas, amplitude, displacement, elevation)
        ax.plot(thetas, fit, color="black")

    ax.plot(angles, scores, color="red", marker="o")
    # ax.scatter(self.angles, self.scores, marker="o", color="black")
    if incl_disp:
        ax.axvline(displacement, color="black", linestyle="--")
        ax.text(
            displacement + 5,
            elevation,
            f"d = {int(displacement)}",
        )
    if incl_amp:
        ax.axhline(amplitude + elevation, color="black", linestyle="--")
        ax.text(0, amplitude + elevation * 0.9, f"a = {amplitude:.2f}")

    if incl_fit:
        ax.text(0, elevation * 0.5, f"R2 = {r2:.2f}")

    ax.set_xticks(OCTANTS)
    ax.set_xticklabels(
        ["0", "45", "90", "135", "180", "225", "270", "315"], fontsize=14
    )
    ax.set_xlabel("Angle [deg]", fontsize=16)
    ax.set_ylabel("Score", fontsize=16)
    ax.set_title(f"{label} Profile", fontsize=20)
    return fig, ax


def sort_angles(angles, scores):
    """Sort angles, scores, and scales in order of scale."""
    # Zip the values and labels together
    zipped = list(zip(angles, scores))
    # Sort the list of zipped tuples
    zipped.sort(key=lambda x: x[0])
    # Unzip the list of tuples
    angles, scores = zip(*zipped)
    return angles, scores
