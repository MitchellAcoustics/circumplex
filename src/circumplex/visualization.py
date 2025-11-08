from __future__ import annotations

from typing import List, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Import only the utils module to avoid circular imports
import circumplex.utils as utils

# Forward type hint for SSMResults to avoid circular imports
# This is needed because ssm_results imports visualization
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from circumplex.ssm_results import SSMResults


def ssm_plot(ssm_object: "SSMResults", fontsize: int = 12, **kwargs) -> Figure:
    """
    Create a figure from SSM results.

    Creates either a circular plot or a contrast plot depending on the
    type of results in ssm_object.

    Parameters
    ----------
    ssm_object : SSMResults
        The results output of ssm_analyze.
    fontsize : int, optional
        Font size of text in the figure, in points.
        Default is 12.
    **kwargs
        Additional arguments to pass on to the plotting function.

    Returns
    -------
    matplotlib.figure.Figure
        A figure object representing the plot.

    Examples
    --------
    >>> from circumplex import ssm_analyze, ssm_plot
    >>> results = ssm_analyze(data, scales, angles)
    >>> fig = ssm_plot(results)
    >>> fig.savefig('ssm_plot.png', dpi=300)
    """
    # Input validation
    if not hasattr(ssm_object, "details") or not hasattr(ssm_object, "results"):
        raise ValueError("ssm_object must be an SSMResults instance")
    if not isinstance(fontsize, (int, float)) or fontsize <= 0:
        raise ValueError("fontsize must be a positive number")

    sns.set(style="whitegrid", font_scale=fontsize / 12)

    if ssm_object.details["contrast"] == "test":
        return ssm_plot_contrast(ssm_object, fontsize=fontsize, **kwargs)
    else:
        return ssm_plot_circle(ssm_object, fontsize=fontsize, **kwargs)


def ssm_plot_circle(
    ssm_object: "SSMResults",
    amax: Optional[float] = None,
    legend_font_size: int = 12,
    scale_font_size: int = 12,
    lowfit: bool = True,
    repel: bool = False,
    angle_labels: Optional[List[str]] = None,
    palette: Optional[str] = "husl",
    **kwargs,
) -> Figure:
    """
    Create a Circular Plot of SSM Results using Seaborn.

    Args:
        ssm_object (SSMResults): The output of ssm_analyze.
        amax (Optional[float]): A positive number corresponding to the radius of the circle.
        legend_font_size (int): Size of the text labels in the legend.
        scale_font_size (int): Size of the text labels for the amplitude and displacement scales.
        lowfit (bool): Whether profiles with low model fit (<.70) should be plotted with dashed borders.
        repel (bool): Experimental argument for plotting text labels instead of colors.
        angle_labels (Optional[List[str]]): Text labels to plot around the circle for each scale.
        palette (Optional[str]): Color palette to use for the plot.
        **kwargs: Additional arguments for seaborn.

    Returns:
        matplotlib.figure.Figure: A figure object containing the circular plot.
    """
    df = ssm_object.results
    angles = np.round(ssm_object.details["angles"]).astype(int)

    if amax is None:
        amax = df["a_uci"].abs().max() * 1.5

    # Convert results to numbers usable by seaborn
    df_plot = df.copy()
    df_plot["d_uci"] = np.where(
        df_plot["d_uci"] < df_plot["d_lci"], df_plot["d_uci"] + 360, df_plot["d_uci"]
    )
    df_plot["a_lci"] = df_plot["a_lci"] * 5 / amax
    df_plot["a_uci"] = df_plot["a_uci"] * 5 / amax
    df_plot["x_est"] = df_plot["x_est"] * 5 / amax
    df_plot["y_est"] = df_plot["y_est"] * 5 / amax

    # Remove profiles with low model fit (unless overridden)
    if not lowfit:
        df_plot = df_plot[df_plot["fit_est"] >= 0.70]
        if len(df_plot) < 1:
            raise ValueError("After removing profiles, there were none left to plot.")

    df_plot["linestyle"] = np.where(df_plot["fit_est"] >= 0.70, "solid", "dashed")

    fig, ax = plt.subplots(figsize=(10, 10))
    circle_base(ax, angles, amax, fontsize=scale_font_size, labels=angle_labels)

    # Use seaborn color palette
    colors = sns.color_palette(palette, n_colors=len(df_plot))

    # Plot confidence regions
    for i, row in df_plot.iterrows():
        wedge = patches.Wedge(
            (0, 0),
            row["a_uci"],
            row["d_lci"],
            row["d_uci"],
            width=row["a_uci"] - row["a_lci"],
            fc=list(colors[i]) + [0.5],  # face opacity at 0.5
            linestyle=row["linestyle"],
            edgecolor=list(colors[i]) + [1],  # edge opacity at 1
            linewidth=2,
        )
        ax.add_patch(wedge)

    # Plot points
    sns.scatterplot(
        data=df_plot,
        x="x_est",
        y="y_est",
        hue="label",
        palette=palette,
        s=100,
        ax=ax,
        legend="brief",
    )

    if repel:
        for _, row in df_plot.iterrows():
            ax.annotate(
                row["label"],
                (row["x_est"], row["y_est"]),
                xytext=(-25 - row["x_est"], 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=legend_font_size,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )
        ax.legend().remove()
    else:
        ax.legend(
            title=ssm_object.details["results_type"],
            fontsize=legend_font_size,
            title_fontsize=legend_font_size,
        )

    return fig


def circle_base(ax, angles, amax=0.5, amin=0, fontsize=12, labels=None):
    """
    Create an Empty Circular Plot.

    Args:
        ax (matplotlib.axes.Axes): The axes to draw on.
        angles (List[float]): Angular displacement of each circumplex scale.
        amax (float): Maximum amplitude.
        amin (float): Minimum amplitude.
        fontsize (int): Font size for labels.
        labels (Optional[List[str]]): Labels for the angles.

    Returns:
        None
    """
    if labels is None:
        labels = [f"{angle}°" for angle in angles]

    # Draw circles
    for r in range(1, 6):
        ax.add_artist(plt.Circle((0, 0), r, fill=False, color="gray"))

    # Draw lines
    for angle in np.deg2rad(angles):
        ax.plot(
            [0, 5 * np.cos(angle)], [0, 5 * np.sin(angle)], color="gray", linewidth=0.5
        )

    # Add labels
    for angle, label in zip(np.deg2rad(angles), labels):
        ax.text(
            5.1 * np.cos(angle),
            5.1 * np.sin(angle),
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
        )

    # Add amplitude labels
    ax.text(
        2,
        0,
        f"{amin + (amax - amin) / 3:.2f}",
        ha="left",
        va="bottom",
        fontsize=fontsize,
    )
    ax.text(
        4,
        0,
        f"{amin + 2 * (amax - amin) / 3:.2f}",
        ha="left",
        va="bottom",
        fontsize=fontsize,
    )

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")


def ssm_plot_contrast(
    ssm_object: "SSMResults",
    axislabel: str = "Difference",
    xy: bool = True,
    color: str = "red",
    linesize: float = 1.25,
    fontsize: int = 12,
) -> Figure:
    """
    Create a Difference Plot of SSM Contrast Results using Seaborn.

    Args:
        ssm_object (SSMResults): The results output of ssm_analyze.
        axislabel (str): Label for the y-axis.
        xy (bool): Whether to include X-Value and Y-Value parameters in the plot.
        color (str): Color of the point range.
        linesize (float): Size of the point range elements in points.
        fontsize (int): Size of the axis labels, numbers, and facet headings in points.

    Returns:
        matplotlib.figure.Figure: A figure object containing the difference plot.
    """
    plabs = {
        "e": r"$\Delta$ Elevation",
        "x": r"$\Delta$ X-Value",
        "y": r"$\Delta$ Y-Value",
        "a": r"$\Delta$ Amplitude",
        "d": r"$\Delta$ Displacement",
    }

    pvals = ["e", "x", "y", "a", "d"]

    res = ssm_object.results

    if not xy:
        res = res.drop(columns=["x_est", "x_lci", "x_uci", "y_est", "y_lci", "y_uci"])
        plabs = {k: v for k, v in plabs.items() if k not in ["x", "y"]}
        pvals = [p for p in pvals if p not in ["x", "y"]]

    res["d_est"] = res["d_est"].astype(float)
    res["d_uci"] = np.where(
        (res["d_uci"] < res["d_lci"]) & (res["d_uci"] < 180),
        (res["d_uci"] + 360) % 360,
        res["d_uci"],
    )
    res["d_lci"] = np.where(
        (res["d_lci"] > res["d_uci"]) & (res["d_lci"] > 180),
        (res["d_lci"] - 360) % 360,
        res["d_lci"],
    )

    # Reshape data for seaborn
    plot_data = []
    for param in pvals:
        for _, row in res.iterrows():
            plot_data.append(
                {
                    "Parameter": plabs[param],
                    "Estimate": row[f"{param}_est"],
                    "Lower CI": row[f"{param}_lci"],
                    "Upper CI": row[f"{param}_uci"],
                    "Label": row["label"],
                }
            )
    plot_df = pd.DataFrame(plot_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.pointplot(
        data=plot_df,
        x="Parameter",
        y="Estimate",
        hue="Label",
        join=False,
        ci=None,
        color=color,
        scale=0.75,
        ax=ax,
    )

    # Add error bars
    ax.errorbar(
        x=plot_df["Parameter"],
        y=plot_df["Estimate"],
        yerr=[
            plot_df["Estimate"] - plot_df["Lower CI"],
            plot_df["Upper CI"] - plot_df["Estimate"],
        ],
        fmt="none",
        ecolor=color,
        elinewidth=linesize,
        capsize=5,
    )

    # Customize plot
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=linesize)
    ax.set_ylabel(axislabel, fontsize=fontsize)
    ax.set_xlabel("")
    ax.tick_params(axis="both", which="major", labelsize=fontsize - 2)
    ax.legend(
        title=ssm_object.details["results_type"],
        fontsize=fontsize - 2,
        title_fontsize=fontsize,
    )

    plt.tight_layout()
    return fig


def ssm_profile_plot(
    scores: Union[np.ndarray, List[float]],
    angles: Union[np.ndarray, List[float], Tuple[float, ...]],
    amplitude: float,
    displacement: float,
    elevation: float,
    r2: Optional[float] = None,
    title: str = "SSM Profile",
    reorder_scales: bool = True,
    incl_pred: bool = True,
    incl_fit: bool = True,
    incl_disp: bool = True,
    incl_amp: bool = True,
    incl_elev: bool = False,
    c_scores: str = "red",
    c_fit: str = "black",
    fontsize: int = 12,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot the SSM profile showing scale scores and fitted cosine curve.

    Creates a profile plot with observed scores and the fitted cosine curve,
    optionally showing displacement, amplitude, elevation, and fit statistics.

    Parameters
    ----------
    scores : array-like
        Array of scores for each scale.
    angles : array-like
        Array of angles for each scale (in degrees).
    amplitude : float
        Amplitude of the cosine function.
    displacement : float
        Angular displacement of the cosine function (in degrees).
    elevation : float
        Elevation (mean level) of the cosine function.
    r2 : float, optional
        R-squared value for the fit.
    title : str, optional
        Title of the plot. Default is "SSM Profile".
    reorder_scales : bool, optional
        Whether to reorder scales based on angles. Default is True.
    incl_pred : bool, optional
        Whether to include the predicted fit line. Default is True.
    incl_fit : bool, optional
        Whether to include the R-squared value. Default is True.
    incl_disp : bool, optional
        Whether to include the displacement line. Default is True.
    incl_amp : bool, optional
        Whether to include the amplitude line. Default is True.
    incl_elev : bool, optional
        Whether to include the elevation line. Default is False.
    c_scores : str, optional
        Color for the score points. Default is "red".
    c_fit : str, optional
        Color for the fit line. Default is "black".
    fontsize : int, optional
        Base font size for the plot. Default is 12.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure and axes.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        A tuple containing the figure and axis objects.

    Examples
    --------
    >>> import numpy as np
    >>> from circumplex import ssm_profile_plot
    >>>
    >>> # Create a profile plot
    >>> scores = np.array([0.5, 0.7, 0.3, -0.2, -0.5, -0.3, 0.1, 0.4])
    >>> angles = (0, 45, 90, 135, 180, 225, 270, 315)
    >>> fig, ax = ssm_profile_plot(
    ...     scores=scores,
    ...     angles=angles,
    ...     amplitude=0.6,
    ...     displacement=30,
    ...     elevation=0.1,
    ...     r2=0.85,
    ...     title="Example Profile"
    ... )
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    assert len(scores) == len(angles), "Scores and angles must be the same length."
    # assert 0 <= elevation <= 1, "Elevation must be between 0 and 1."
    if r2 is not None:
        assert 0 <= r2 <= 1, "R2 must be between 0 and 1."
    assert 0 <= amplitude, "Amplitude must be a positive number."
    assert 0 <= displacement <= 360, "Displacement must be between 0 and 360."

    if reorder_scales:
        angles, scores = utils.sort_angles(angles, scores)
        if angles[-1] == 360:
            angles = (0,) + angles
            scores = (scores[-1],) + scores

    if incl_pred:
        thetas = np.linspace(0, 360, 1000)
        fit = utils.cosine_form(
            np.deg2rad(thetas), amplitude, np.deg2rad(displacement), elevation
        )
        ax.plot(thetas, fit, color=c_fit)

    ax.plot(angles, scores, color=c_scores, marker="o")

    if incl_disp:
        ax.axvline(displacement, color="black", linestyle="--")
        ax.text(
            displacement + 5, elevation, f"d = {int(displacement)}", fontsize=fontsize
        )

    if incl_amp:
        ax.axhline(amplitude + elevation, color="black", linestyle="--")
        ax.text(
            0, elevation + amplitude * 0.5, f"a = {amplitude:.2f}", fontsize=fontsize
        )

    if incl_fit:
        ylim = ax.get_ylim()
        ax.text(
            0, ylim[0] + 0.25 * (ylim[1] - ylim[0]), f"R2 = {r2:.2f}", fontsize=fontsize
        )

    if incl_elev:
        ax.axhline(elevation, color="black", linestyle="--")
        ax.text(0, elevation, f"e = {elevation:.2f}", fontsize=fontsize)

    ax.set_xticks(utils.OCTANTS)
    ax.set_xticklabels(
        ["0", "45", "90", "135", "180", "225", "270", "315"], fontsize=fontsize
    )
    ax.set_xlabel("Angle [deg]", fontsize=fontsize + 2)
    ax.set_ylabel("Score", fontsize=fontsize + 2)
    ax.set_title(title, fontsize=fontsize + 4)

    return fig, ax
