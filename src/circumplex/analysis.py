# %%

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

OCTANTS = (0, 45, 90, 135, 180, 225, 270, 315)


class SSMParams(object):
    def __init__(
        self,
        scores,
        scales,
        angles=OCTANTS,
        group=None,
        measure=None,
        bounds=([-np.inf, 0, -np.inf], [np.inf, 360, np.inf]),
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
    def label(self):
        if self.group is not None and self.measure is not None:
            return f"{self.group}_{self.measure}"
        elif self.measure is not None:
            return self.measure
        elif self.group is not None:
            return self.group
        else:
            return "SSM"

    @property
    def table(self):
        scale_angle = {scale: angle for scale, angle in zip(self.scales, self.angles)}
        return pd.DataFrame(
            self.params | scale_angle,
            index=[self.label],
        )

    @property
    def params(self):
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
        return f"SSMParams(scores={self.scores}, angles={self.angles})"

    def __str__(self):
        # TODO: Add param results
        return f"SSMParams(scores={self.scores}, angles={self.angles})"

    def profile_plot(self):
        """Plot the SSM profile."""
        thetas = np.linspace(0, 360, 1000)
        fit = cosine_form(thetas, self.amplitude, self.displacement, self.elevation)

        fig, ax = plt.subplots()
        ax.plot(thetas, fit, color="black")
        ax.plot(self.angles, self.scores, color="red", marker="o")
        # ax.scatter(self.angles, self.scores, marker="o", color="black")
        ax.axvline(self.displacement, color="black", linestyle="--")
        ax.text(
            self.displacement + 5,
            self.elevation,
            f"d = {int(self.displacement)}",
        )
        ax.axhline(self.elevation - self.amplitude, color="black", linestyle="--")
        ax.text(0, self.elevation - self.amplitude * 0.9, f"a = {self.amplitude:.2f}")

        ax.text(0, self.elevation * 0.5, f"R2 = {self.r2:.2f}")

        ax.set_xticks(OCTANTS)
        ax.set_xticklabels(
            ["0", "45", "90", "135", "180", "225", "270", "315"], fontsize=14
        )
        ax.set_xlabel("Angle [deg]", fontsize=16)
        ax.set_ylabel("Score", fontsize=16)
        ax.set_title(f"{self.label} Profile", fontsize=20)
        return fig, ax

    def plot(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.plot(
            np.deg2rad(self.displacement),
            self.amplitude,
            color="black",
            marker="o",
            markersize=10,
        )


class SSMResults(object):
    def __init__(self, results, measures=None, grouping=None):
        self.results = results
        self.measures = measures
        self.grouping = grouping

    @property
    def table(self):
        df = pd.DataFrame()
        for key, val in self.results.items():
            df = pd.concat([df, val.table])
        return df

    def plot(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        for key, val in self.results.items():
            ax.plot(
                np.deg2rad(val.displacement),
                val.amplitude,
                color="black",
                marker="o",
                markersize=10,
            )
        return fig, ax

    def profile_plots(self):
        for key, val in self.results.items():
            fig, ax = val.profile_plot()
            plt.show()


def ssm_analyse(data, scales, measures=None, grouping=None, angles=OCTANTS):
    if grouping is not None and measures is not None:
        return ssm_analyse_grouped_corrs(data, scales, measures, grouping, angles)
    elif measures is not None:
        return ssm_analyse_corrs(data, scales, measures, angles)
    elif grouping is not None:
        return ssm_analyse_means(data, scales, grouping, angles)
    else:
        ssm = SSMParams(data[scales].mean(), scales, angles)
        # ssm.param_calc()
        return ssm


def ssm_analyse_grouped_corrs(data, scales, measures, grouping, angles=OCTANTS):
    res = {}
    for measure in measures:
        for group, group_data in data.groupby(grouping):
            try:
                group = group[0]
                r = group_data[scales].corrwith(group_data[measure])
                ssm = SSMParams(r, scales, angles, group=group, measure=measure)
                # ssm.param_calc()
                res[ssm.label] = ssm
            except:
                print(f"Error in {group} for {measure}")
    return res


def ssm_analyse_corrs(data, scales, measures, angles=OCTANTS):
    res = {}
    for measure in measures:
        r = data[scales].corrwith(data[measure])
        ssm = SSMParams(r, scales, angles, measure=measure)
        # ssm.param_calc()
        res[ssm.label] = ssm

    return res


def ssm_analyse_means(data, scales, grouping, angles=OCTANTS):
    means = data.groupby(grouping)[scales].mean()
    res = {}
    for group, scores in means.iterrows():
        scores = means.loc[group]
        ssm = SSMParams(scores, scales, angles, group=group)
        # ssm.param_calc()
        res[ssm.label] = ssm

    return res


def cosine_form(theta, ampl, disp, elev):
    """Cosine function with amplitude, dispersion and elevation parameters."""
    return elev + ampl * np.cos(np.deg2rad(theta - disp))


def _r2_score(y_true, y_pred):
    """Calculate the R2 score for a set of predictions."""
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - (ss_res / ss_tot)


def ssm_parameters(
    scores, angles, bounds=([-np.inf, 0, -np.inf], [np.inf, 360, np.inf])
):
    """Calculate SSM parameters (without confidence intervals) for a set of scores.

    Args:
        scores (np.array): A numeric vector (or single row dataframe) containing one score for each of a set of circumplex scales.
        angles (tuple): A numeric vector containing the angular displacement of each circumplex scale included in `scores`.
        bounds (tuple, optional): The bounds for each of the parameters of the curve optimisation. Defaults to ([0, 0, -1], [np.inf, 360, 1]).

    Returns:
        tuple: A tuple containing the elevation, x-value, y-value, amplitude, displacement, and R2 fit of the SSM curve.

    Examples:

        >>> scores = np.array([-0.5, 0, 0.25, 0.51, 0.52, 0.05, -0.26, -0.7])
        >>> angles = OCTANTS
        >>> ssm_parameters(scores, angles)
        (0.5, 0.0, 0.0, 0.0, 0.0, 1.0)
    """

    # noinspection PyTupleAssignmentBalance
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
