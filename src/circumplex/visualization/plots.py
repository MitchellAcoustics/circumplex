"""Matplotlib-based plotting functions for SSM results."""

from __future__ import annotations

import warnings
from typing import Any, cast

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from circumplex.visualization._utils import pretty_max


def _validate_results_df(results_df: pd.DataFrame) -> None:
    """Validate that results DataFrame has required columns.

    Parameters
    ----------
    results_df
        DataFrame to validate.

    Raises
    ------
    ValueError
        If required columns are missing.

    """
    required_cols = ["Label", "e_est", "x_est", "y_est", "a_est", "d_est", "fit_est"]
    missing = [col for col in required_cols if col not in results_df.columns]
    if missing:
        msg = f"results_df missing required columns: {missing}"
        raise ValueError(msg)


def _prepare_plot_data(
    results_df: pd.DataFrame,
    profile_indices: list[int] | None,
    *,
    drop_lowfit: bool,
) -> pd.DataFrame:
    """Extract and filter profiles to plot.

    Parameters
    ----------
    results_df
        Full results DataFrame.
    profile_indices
        Indices of profiles to plot, or None for all.
    drop_lowfit
        Whether to drop profiles with fit < 0.70.

    Returns
    -------
    Filtered DataFrame ready for plotting.

    Raises
    ------
    IndexError
        If profile index is out of range.
    ValueError
        If no profiles remain after filtering.

    """
    # Determine which profiles to plot
    if profile_indices is None:
        profile_indices = list(range(len(results_df)))

    if len(profile_indices) == 0:
        msg = "No profiles to plot"
        raise ValueError(msg)

    # Validate indices
    for idx in profile_indices:
        if idx >= len(results_df):
            msg = f"Profile index {idx} out of range (max: {len(results_df) - 1})"
            raise IndexError(msg)

    # Extract profiles
    plot_df = results_df.iloc[profile_indices].copy()

    # Handle low-fit profiles
    if drop_lowfit:
        low_fit_mask = plot_df["fit_est"] < 0.70
        if low_fit_mask.any():
            dropped = plot_df.loc[low_fit_mask, "Label"].tolist()
            warnings.warn(
                f"Dropping profiles with fit < 0.70: {dropped}",
                stacklevel=3,
            )
            plot_df = plot_df[~low_fit_mask]
            if len(plot_df) == 0:
                msg = "All profiles dropped due to low fit"
                raise ValueError(msg)

    # Add line style column based on fit
    plot_df["linestyle"] = plot_df["fit_est"].apply(
        lambda fit: "solid" if fit >= 0.70 else "dashed"
    )

    return plot_df


def _setup_colors(
    n_profiles: int,
    colors: str | list[str] | None,
) -> tuple[list[tuple[float, float, float]], bool]:
    """Set up colors and determine if legend should be shown.

    Parameters
    ----------
    n_profiles
        Number of profiles to plot.
    colors
        Color specification. Can be:
        - None: single blue color, no legend
        - str: seaborn palette name (e.g., "Set2", "husl")
        - list: custom colors as names, hex codes, or RGB tuples

    Returns
    -------
    tuple of (colors list, show_legend boolean)

    Raises
    ------
    ValueError
        If custom color list is provided but empty.

    """
    # No colors specified or single profile with default
    if colors is None or (colors == "Set2" and n_profiles == 1):
        return [(0.0, 0.45, 0.70)] * n_profiles, False

    # Palette name (string)
    if isinstance(colors, str):
        if n_profiles > 1:
            color_list = sns.color_palette(colors, n_profiles)
            return color_list, True
        # Single profile with named palette
        color_list = sns.color_palette(colors, 1)
        return color_list, False

    # Custom color list
    if len(colors) == 0:
        msg = "colors list cannot be empty"
        raise ValueError(msg)

    # Convert all colors to RGB tuples
    color_list: list[tuple[float, float, float]] = []
    for color in colors:
        try:
            color_list.append(to_rgb(color))
        except ValueError as e:
            msg = f"Invalid color specification: {color}"
            raise ValueError(msg) from e

    # Cycle colors if not enough provided
    if len(color_list) < n_profiles:
        warnings.warn(
            f"Only {len(color_list)} colors provided for {n_profiles} profiles. "
            f"Colors will be cycled.",
            stacklevel=4,
        )
        # Repeat colors to cover all profiles
        multiplier = (n_profiles // len(color_list)) + 1
        color_list = (color_list * multiplier)[:n_profiles]

    return color_list, n_profiles > 1


def _get_contrast_row(results_df: pd.DataFrame) -> pd.Series:
    """Get the contrast row (last row) from results_df."""
    if len(results_df) < 3:
        msg = (
            "Contrast plot requires at least 3 rows in results_df "
            "(two profiles + contrast)"
        )
        raise ValueError(msg)
    return results_df.iloc[-1]


def _build_contrast_plot_data(
    contrast_row: pd.Series, *, drop_xy: bool
) -> list[dict[str, Any]]:
    """Build per-parameter contrast plot data from a results row.

    Parameters
    ----------
    contrast_row
        The contrast row from results_df (typically last row).
    drop_xy
        Whether to omit X and Y parameters.

    Returns
    -------
    List of dictionaries with keys: parameter, estimate, lci, uci, significant.
    """
    param_names = ["e", "x", "y", "a", "d"]
    param_labels = [
        "Δ Elevation",
        "Δ X-Value",
        "Δ Y-Value",
        "Δ Amplitude",
        "Δ Displacement",
    ]

    plot_data: list[dict[str, Any]] = []
    for param, label in zip(param_names, param_labels, strict=False):
        est_col = f"{param}_est"
        lci_col = f"{param}_lci"
        uci_col = f"{param}_uci"

        if est_col not in contrast_row:
            msg = f"Missing required column: {est_col}"
            raise ValueError(msg)

        lci = contrast_row[lci_col]
        uci = contrast_row[uci_col]
        significant = not (lci <= 0 <= uci)

        plot_data.append(
            {
                "parameter": label,
                "estimate": contrast_row[est_col],
                "lci": lci,
                "uci": uci,
                "significant": significant,
            }
        )

    if drop_xy:
        plot_data = [
            d for d in plot_data if d["parameter"] not in ["Δ X-Value", "Δ Y-Value"]
        ]

    return plot_data


def _draw_contrast_subplot(
    *,
    ax: plt.Axes,
    data: dict[str, Any],
    sig_color: str,
    ns_color: str,
    linewidth: float,
    fontsize: float,
    is_leftmost: bool,
) -> None:
    """Render a single contrast subplot."""
    # Draw horizontal line at zero
    ax.axhline(0, color="darkgray", linewidth=linewidth * 0.8, zorder=1)

    # Determine color based on significance
    color = sig_color if data["significant"] else ns_color

    # Draw error bar
    lower_error = abs(data["estimate"] - data["lci"])
    upper_error = abs(data["uci"] - data["estimate"])
    ax.errorbar(
        0,
        data["estimate"],
        yerr=[[lower_error], [upper_error]],
        fmt="none",
        ecolor="black",
        elinewidth=linewidth,
        capsize=5,
        capthick=linewidth,
        zorder=2,
    )

    # Draw point
    ax.scatter(
        0,
        data["estimate"],
        s=linewidth * 120,
        c=[color],
        edgecolors="black",
        linewidths=linewidth,
        zorder=3,
    )

    # Set title
    ax.set_title(data["parameter"], fontsize=fontsize * 1.05, pad=10)

    # Remove x-axis
    ax.set_xlim(-0.4, 0.4)
    ax.set_xticks([])

    # Style y-axis - only show ylabel on leftmost plot
    if is_leftmost:
        ax.set_ylabel("Difference", fontsize=fontsize, fontweight="bold")
    ax.tick_params(axis="y", labelsize=fontsize * 0.9)

    # Apply seaborn style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)


def _add_contrast_legend(
    fig: Figure, sig_color: str, ns_color: str, fontsize: float
) -> None:
    """Add a legend to the contrast plot."""
    legend_elements = [
        Patch(
            facecolor=sig_color,
            edgecolor="black",
            linewidth=1.5,
            label="Significant (p < .05)",
        ),
        Patch(
            facecolor=ns_color,
            edgecolor="black",
            linewidth=1.5,
            label="Not Significant",
        ),
    ]
    legend = fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=True,
        fontsize=fontsize * 0.95,
        edgecolor="gray",
        fancybox=False,
    )
    legend.get_frame().set_linewidth(1.2)


def _draw_circle_base(
    ax: plt.Axes,
    angles: list[float] | np.ndarray,
    amax: float,
    fontsize: float = 12,
    labels: list[str] | None = None,
) -> None:
    """Draw the base circumplex circle with scales.

    Parameters
    ----------
    ax
        Matplotlib axes to draw on.
    angles
        Angular positions in degrees for displacement scale.
    amax
        Maximum amplitude value for scale.
    fontsize
        Font size for labels in points.
    labels
        Labels for each angle. If None, uses degree symbols (e.g., "90°").

    """
    # Default labels: degree symbols
    if labels is None:
        labels = [f"{int(angle)}°" for angle in angles]
    elif len(labels) != len(angles):
        msg = f"labels must have same length as angles ({len(angles)})"
        raise ValueError(msg)

    # Set up axes
    ax.set_aspect("equal")
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-6.5, 6.5)
    ax.axis("off")

    # Draw outer circle (radius 5)
    outer_circle = mpatches.Circle(
        (0, 0),
        5,
        fill=True,
        facecolor="white",
        edgecolor="gray",
        linewidth=1.5,
    )
    ax.add_patch(outer_circle)

    # Draw radial segments for displacement scale
    angles_rad = np.radians(angles)
    for angle_rad in angles_rad:
        x_end = 5 * np.cos(angle_rad)
        y_end = 5 * np.sin(angle_rad)
        ax.plot([0, x_end], [0, y_end], color="gray", linewidth=0.5, alpha=0.6)

    # Draw amplitude circles (radii 1-4)
    for radius in range(1, 5):
        circle = mpatches.Circle(
            (0, 0),
            radius,
            fill=False,
            edgecolor="gray",
            linewidth=0.5,
            alpha=0.6,
        )
        ax.add_patch(circle)

    # Draw amplitude scale labels (at positions 2 and 4 on x-axis)
    amp_values = np.linspace(0, amax, 6)
    for radius, amp_val in zip([2, 4], [amp_values[2], amp_values[4]], strict=False):
        ax.text(
            radius,
            0,
            f"{amp_val:.2f}",
            ha="center",
            va="center",
            fontsize=fontsize * 0.8,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "none",
            },
        )

    # Draw displacement scale labels (at radius 5.1)
    for label, angle_rad in zip(labels, angles_rad, strict=False):
        x_label = 5.1 * np.cos(angle_rad)
        y_label = 5.1 * np.sin(angle_rad)

        # Determine alignment based on position
        if np.abs(x_label) < 0.5:
            ha = "center"
        elif x_label > 0:
            ha = "left"
        else:
            ha = "right"

        if np.abs(y_label) < 0.5:
            va = "center"
        elif y_label > 0:
            va = "bottom"
        else:
            va = "top"

        ax.text(
            x_label,
            y_label,
            label,
            ha=ha,
            va=va,
            fontsize=fontsize * 0.8,
            color="gray",
        )


def plot_circle(
    results_df: pd.DataFrame,
    angles: list[float] | np.ndarray,
    *,
    profile_indices: list[int] | None = None,
    amax: float | None = None,
    angle_labels: list[str] | None = None,
    colors: str | list[str] | None = "Set2",
    fontsize: float = 12,
    drop_lowfit: bool = False,
    figsize: tuple[float, float] = (8, 8),
    title: str | None = None,
) -> Figure:
    """Plot SSM profiles on a circumplex circle.

    Creates a circular plot showing amplitude and displacement of SSM profiles,
    with arc bars representing confidence intervals. Automatically handles both
    single and multiple profiles.

    Parameters
    ----------
    results_df
        DataFrame with SSM results. Must contain columns:
        - Label: str, profile name
        - e_est, x_est, y_est, a_est, d_est: float, parameter estimates
        - e_lci, x_lci, ..., d_uci: float, confidence intervals
        - fit_est: float, model fit (0-1)
    angles
        Angular positions of scales in degrees
        (e.g., [90, 135, 180, 225, 270, 315, 360, 45]).
    profile_indices
        Which rows of results_df to plot. If None, plots all profiles.
    amax
        Maximum amplitude for scaling. If None, auto-computed using pretty_max().
    angle_labels
        Labels for each angle. If None, shows degree symbols (e.g., "90°").
        Pass empty strings to hide labels.
    colors
        Colors for profiles. Can be:
        - Seaborn palette name: "Set2", "husl", "deep", etc.
        - List of color specifications: ['red', 'blue'] or ['#FF0000', '#0000FF']
        - None: single blue color with no legend
    fontsize
        Base font size in points.
    drop_lowfit
        If True, omit profiles with fit < 0.70. If False, show with dashed borders.
    figsize
        Figure size in inches (width, height).
    title
        Title for the plot. If None, no title is added.

    Returns
    -------
    Matplotlib Figure object.

    Examples
    --------
    Plot all profiles from an SSM analysis:

    >>> from circumplex import ssm_analyze
    >>> results = ssm_analyze(data, scales=list(range(8)))
    >>> fig = plot_circle(results.results, results.details.angles)
    >>> fig.savefig('profiles.png')

    Plot specific profiles with custom styling:

    >>> fig = plot_circle(
    ...     results.results,
    ...     results.details.angles,
    ...     profile_indices=[0, 1],
    ...     colors="husl",
    ...     fontsize=14,
    ...     figsize=(10, 10),
    ... )

    Use custom colors:

    >>> fig = plot_circle(
    ...     results.results,
    ...     results.details.angles,
    ...     colors=['red', 'blue', 'green'],
    ... )

    See Also
    --------
    plot_curve : Plot SSM fitted curves with observed scores
    plot_contrast : Plot SSM parameter contrasts

    """
    # Validate and prepare data
    _validate_results_df(results_df)
    plot_df = _prepare_plot_data(results_df, profile_indices, drop_lowfit=drop_lowfit)

    # Determine amax
    if amax is None:
        if "a_uci" in plot_df.columns:
            amax = pretty_max(plot_df["a_uci"].values)
        else:
            amax = pretty_max(plot_df["a_est"].values)

    # Create figure and draw base
    fig, ax = plt.subplots(figsize=figsize)
    _draw_circle_base(ax, angles, amax, fontsize, angle_labels)

    # Scale factor: radius 5 corresponds to amax
    scale_factor = 5.0 / amax

    # Set up colors and legend
    n_profiles = len(plot_df)
    color_list, show_legend = _setup_colors(n_profiles, colors)

    # Plot each profile
    for i, (_idx, row) in enumerate(plot_df.iterrows()):
        color = color_list[i]
        label = row["Label"]

        # Scale parameters to circle coordinates
        x_plot = row["x_est"] * scale_factor
        y_plot = row["y_est"] * scale_factor

        # Draw confidence interval arc if available
        ci_cols = ["a_lci", "a_uci", "d_lci", "d_uci"]
        has_ci = all(col in row and not pd.isna(row[col]) for col in ci_cols)

        if has_ci:
            a_lci_plot = row["a_lci"] * scale_factor
            a_uci_plot = row["a_uci"] * scale_factor
            d_lci = row["d_lci"]
            d_uci = row["d_uci"]

            # Handle displacement wrapping (CI crosses 0/360)
            if d_uci < d_lci:
                d_uci += 360

            # Draw arc bar (wedge)
            wedge = mpatches.Wedge(
                center=(0, 0),
                r=a_uci_plot,
                theta1=d_lci,
                theta2=d_uci,
                width=a_uci_plot - a_lci_plot,
                facecolor=color,
                edgecolor=color,
                alpha=0.4,
                linestyle=row["linestyle"],
                linewidth=1.0,
            )
            ax.add_patch(wedge)

        # Draw point at (x, y)
        ax.scatter(
            x_plot,
            y_plot,
            s=100,
            facecolor=color,
            edgecolor="black",
            linewidth=1.0,
            zorder=10,
            label=label if show_legend else None,
        )

    # Add legend if multiple profiles
    if show_legend:
        ax.legend(
            title="Profile",
            fontsize=fontsize * 0.9,
            title_fontsize=fontsize,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            frameon=True,
        )

    # Add title if provided
    if title is not None:
        fig.suptitle(title, fontsize=fontsize * 1.2, fontweight="bold")

    plt.tight_layout()

    return fig


def _plot_single_curve(
    ax: plt.Axes,
    angles: np.ndarray,
    angles_sorted: np.ndarray,
    observed_scores: np.ndarray,
    scores_sorted: np.ndarray,
    result_row: pd.Series,
    c_scores: str,
    c_fit: str,
    tick_labels: list[str],
    xlabel: str,
    base_size: float,
    *,
    incl_pred: bool,
    incl_fit: bool,
    incl_disp: bool,
    incl_amp: bool,
    incl_elev: bool,
) -> None:
    """Plot a single SSM curve on an axes.

    Parameters
    ----------
    ax
        Matplotlib axes to plot on.
    angles
        Angular positions (original order).
    angles_sorted
        Angular positions (sorted order).
    observed_scores
        Observed scores (original order).
    scores_sorted
        Observed scores (sorted order).
    result_row
        Row from results DataFrame with SSM parameters.
    color
        RGB color tuple for the fitted curve.
    tick_labels
        Labels for x-axis ticks.
    xlabel
        Label for x-axis.
    base_size
        Base font size.

    """
    # Plot observed scores as points (at original positions)
    ax.plot(
        angles,
        observed_scores,
        "o",
        color=c_scores,
        # markersize=6,  # noqa: ERA001
        # zorder=3,  # noqa: ERA001
        label="Observed",
    )

    # Connect observed scores with lines (using sorted order)
    ax.plot(
        angles_sorted,
        scores_sorted,
        "-",
        color=c_scores,
        # linewidth=0.8,  # noqa: ERA001
        # alpha=0.5,  # noqa: ERA001
        zorder=2,
    )

    # Generate fitted curve
    amplitude = result_row["a_est"]
    displacement = result_row["d_est"]
    elevation = result_row["e_est"]
    r2 = result_row["fit_est"]

    if incl_pred:
        angle_range = np.linspace(min(angles), max(angles), 100)
        fitted_scores = elevation + amplitude * np.cos(
            np.radians(angle_range - displacement)
        )

        # Determine line style based on fit
        linestyle = "solid" if r2 >= 0.70 else "dashed"

        # Plot fitted curve
        ax.plot(
            angle_range,
            fitted_scores,
            linestyle=linestyle,
            color=c_fit,
            # linewidth=2.0,  # noqa: ERA001
            # zorder=4,  # noqa: ERA001
            label="Fitted",
        )

    # Annotate parameters
    ymin, ymax = ax.get_ylim()
    y_offset = (ymax - ymin) * 0.2
    curve_min = displacement - 180 if displacement >= 180 else displacement + 180

    if incl_disp:
        ax.axvline(displacement, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(
            displacement,
            max(fitted_scores) - y_offset,
            f"d = {int(displacement)}°",
            horizontalalignment="center",
            verticalalignment="center",
            bbox={"facecolor": "white"},
            fontsize=base_size * 0.8,
        )
    if incl_amp:
        ax.axhline(
            amplitude + elevation,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )
        ax.text(
            curve_min,
            elevation + amplitude,
            f"a = {amplitude:.2f}",
            horizontalalignment="center",
            verticalalignment="center",
            bbox={"facecolor": "white"},
            fontsize=base_size * 0.8,
        )
    if incl_fit:
        ax.text(
            curve_min,
            elevation + (amplitude * 0.5),
            f"R² = {r2:.2f}",
            horizontalalignment="center",
            verticalalignment="center",
            bbox={"facecolor": "white"},
            fontsize=base_size * 0.8,
        )
    if incl_elev:
        ax.axhline(elevation, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(
            curve_min,
            elevation,
            f"e = {elevation:.2f}",
            horizontalalignment="center",
            verticalalignment="center",
            bbox={"facecolor": "white"},
            fontsize=base_size * 0.8,
        )

    # Set x-axis ticks and labels
    ax.set_xticks(angles)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    # Add title with profile label
    ax.set_title(result_row["Label"], fontsize=base_size * 1.1, fontweight="bold")

    # Add axis labels
    ax.set_xlabel(xlabel, fontsize=base_size)
    ax.set_ylabel("Score", fontsize=base_size)

    # Apply seaborn style
    ax.grid(axis="x", alpha=0.5, linestyle="-", linewidth=0.5)
    ax.grid(axis="y", alpha=0.5, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)


def plot_curve(
    results_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    angles: list[float] | np.ndarray,
    *,
    profile_indices: list[int] | None = None,
    angle_labels: list[str] | None = None,
    c_scores: str = "red",
    c_fit: str = "black",
    base_size: float = 11,
    drop_lowfit: bool = False,
    figsize: tuple[float, float] | None = None,
    incl_pred: bool = True,
    incl_fit: bool = False,
    incl_disp: bool = False,
    incl_amp: bool = False,
    incl_elev: bool = False,
) -> Figure:
    """Plot SSM fitted curves with observed scores.

    Creates a faceted plot showing the fitted cosine curve overlaid on the
    observed circumplex scale scores. Each profile is shown in a separate subplot.

    Parameters
    ----------
    results_df
        DataFrame with SSM results. Must contain columns:
        - Label: str, profile name
        - e_est, a_est, d_est: float, parameter estimates
        - fit_est: float, model fit (0-1)
    scores_df
        DataFrame with observed circumplex scores. Must have columns:
        - Label: str, profile name (matching results_df)
        - Scale columns: float, one column per circumplex scale
    angles
        Angular positions of scales in degrees, matching score column order.
    profile_indices
        Which rows to plot. If None, plots all profiles.
    angle_labels
        Labels for each angle on x-axis. If None, shows degree symbols (e.g., "90°").
    c_scores
        Color for observed scores (points and lines).
    c_fit
        Color for fitted curve.
    base_size
        Base font size in points for labels and text.
    drop_lowfit
        If True, omit profiles with fit < 0.70. If False, show with dashed curves.
    figsize
        Figure size in inches (width, height). If None, auto-computed based on
        number of profiles.

    Returns
    -------
    Matplotlib Figure object.

    Examples
    --------
    Plot curves from an SSM analysis:

    >>> from circumplex import ssm_analyze
    >>> results = ssm_analyze(data, scales=list(range(8)))
    >>> fig = plot_curve(results.results, results.scores, results.details.angles)
    >>> fig.savefig('curves.png')

    Use custom angle labels:

    >>> fig = plot_curve(
    ...     results.results,
    ...     results.scores,
    ...     results.details.angles,
    ...     angle_labels=['PA', 'BC', 'DE', 'FG', 'HI', 'JK', 'LM', 'NO'],
    ... )

    See Also
    --------
    plot_circle : Plot SSM profiles on a circumplex circle
    plot_contrast : Plot SSM parameter contrasts

    """
    # Validate results
    _validate_results_df(results_df)

    # Prepare plot data (filter profiles, handle low fit)
    plot_results = _prepare_plot_data(
        results_df, profile_indices, drop_lowfit=drop_lowfit
    )

    # Filter scores to match plot_results
    plot_scores = scores_df[scores_df["Label"].isin(plot_results["Label"])].copy()

    n_profiles = len(plot_results)
    if n_profiles == 0:
        msg = "No profiles to plot"
        raise ValueError(msg)

    # Determine subplot layout
    ncols = min(3, n_profiles)
    nrows = (n_profiles + ncols - 1) // ncols

    # Auto-compute figure size if not provided
    if figsize is None:
        figsize = (ncols * 6, nrows * 4)

    # Set up colors
    # color_list, _ = _setup_colors(n_profiles, colors)  # noqa: ERA001

    # Prepare angle labels for x-axis
    if angle_labels is None:
        xlabel = "Angle"
        tick_labels = [f"{int(a)}°" for a in angles]
    else:
        xlabel = "Scale"
        if len(angle_labels) != len(angles):
            msg = f"angle_labels must have same length as angles ({len(angles)})"
            raise ValueError(msg)
        tick_labels = angle_labels

    # Create figure with subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    # Convert ndarray of Axes to a plain list with proper typing for type checkers
    axes_flat: list[plt.Axes] = [cast("plt.Axes", a) for a in axes.flatten()]

    # Hide unused subplots
    for idx in range(n_profiles, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Get scale column names (everything except Label, Model, Fit)
    info_cols = {"Label", "Model", "Fit"}
    scale_cols = [col for col in plot_scores.columns if col not in info_cols]

    if len(scale_cols) != len(angles):
        msg = (
            f"Number of scale columns ({len(scale_cols)}) must match "
            f"angles ({len(angles)})"
        )
        raise ValueError(msg)

    # Prepare angles array for sorting
    angles_array = np.array(angles)

    # Plot each profile
    for idx, (_ridx, result_row) in enumerate(plot_results.iterrows()):
        ax = axes_flat[idx]
        label = result_row["Label"]

        # Get scores for this profile
        score_row = plot_scores[plot_scores["Label"] == label].iloc[0]
        observed_scores = score_row[scale_cols].to_numpy().astype(float)

        # Sort angles and scores together for proper line connection
        sorted_indices = np.argsort(angles_array)
        angles_sorted = angles_array[sorted_indices]
        scores_sorted = observed_scores[sorted_indices]

        # Plot using helper function
        _plot_single_curve(
            ax,
            angles_array,
            angles_sorted,
            observed_scores,
            scores_sorted,
            result_row,
            c_scores,
            c_fit,
            tick_labels,
            xlabel,
            base_size,
            incl_pred=incl_pred,
            incl_fit=incl_fit,
            incl_disp=incl_disp,
            incl_amp=incl_amp,
            incl_elev=incl_elev,
        )

    plt.tight_layout()

    return fig


def plot_contrast(
    results_df: pd.DataFrame,
    *,
    drop_xy: bool = False,
    sig_color: str = "#fc8d62",
    ns_color: str = "white",
    linewidth: float = 1.25,
    fontsize: float = 12,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot SSM parameter contrasts with confidence intervals.

    Creates a faceted plot showing the difference between two profiles for each
    SSM parameter (elevation, x-value, y-value, amplitude, displacement). Points
    are colored based on statistical significance (whether CI includes zero).

    This function requires results from a contrast analysis (e.g., comparing two
    groups or measures).

    Parameters
    ----------
    results_df
        DataFrame with SSM contrast results. Must contain the contrast row
        (typically the last row) with columns:
        - Label: str, contrast label (e.g., "Group 1 - Group 2")
        - e_est, x_est, y_est, a_est, d_est: float, parameter differences
        - e_lci, x_lci, ..., d_uci: float, confidence intervals
    drop_xy
        Whether to omit x-value and y-value parameters from the plot. This can
        simplify the plot when only interested in elevation, amplitude, and
        displacement (default = False).
    sig_color
        Color for significant contrasts (CI excludes zero).
    ns_color
        Color for non-significant contrasts (CI includes zero).
    linewidth
        Width of error bars and point outlines in points.
    fontsize
        Base font size in points for labels and text.
    figsize
        Figure size in inches (width, height). If None, uses (10, 4) for all
        parameters or (7, 4) if drop_xy=True.

    Returns
    -------
    Matplotlib Figure object.

    Examples
    --------
    Plot contrasts from an SSM analysis:

    >>> from circumplex import ssm_analyze
    >>> results = ssm_analyze(
    ...     data, scales=list(range(8)),
    ...     grouping='condition', contrast=True
    ... )
    >>> fig = plot_contrast(results.results)
    >>> fig.savefig('contrasts.png')

    Drop x and y parameters for simpler plot:

    >>> fig = plot_contrast(results.results, drop_xy=True)

    Use custom colors:

    >>> fig = plot_contrast(
    ...     results.results,
    ...     sig_color='red',
    ...     ns_color='lightgray',
    ... )

    See Also
    --------
    plot_circle : Plot SSM profiles on a circumplex circle
    plot_curve : Plot SSM fitted curves with observed scores

    """
    # Build data
    contrast_row = _get_contrast_row(results_df)
    plot_data = _build_contrast_plot_data(contrast_row, drop_xy=drop_xy)

    n_params = len(plot_data)

    # Set figure size
    if figsize is None:
        figsize = (7, 4) if drop_xy else (10, 4)

    # Create figure with subplots
    fig, axes = plt.subplots(
        1, n_params, figsize=figsize, sharey=False, constrained_layout=False
    )

    # Normalize axes into a list of Axes for consistent typing/iteration
    if n_params == 1:
        axes_list: list[plt.Axes] = [cast("plt.Axes", axes)]
    else:
        axes_list = [cast("plt.Axes", a) for a in cast("np.ndarray", axes)]

    # Plot each parameter
    for i, (ax, data) in enumerate(zip(axes_list, plot_data, strict=False)):
        _draw_contrast_subplot(
            ax=ax,
            data=data,
            sig_color=sig_color,
            ns_color=ns_color,
            linewidth=linewidth,
            fontsize=fontsize,
            is_leftmost=(i == 0),
        )

    # Add legend with better positioning
    _add_contrast_legend(fig, sig_color, ns_color, fontsize)

    # Add contrast label as suptitle
    contrast_label = contrast_row["Label"]
    fig.suptitle(
        f"Contrast: {contrast_label}",
        fontsize=fontsize * 1.15,
        fontweight="bold",
        y=0.88,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.86])

    return fig
