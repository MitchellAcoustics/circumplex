import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Optional, List
from circumplex.core.ssm_results import SSMResults

def ssm_plot(ssm_object: SSMResults, fontsize: int = 12, **kwargs):
    """
    Create a figure from SSM results.

    Args:
        ssm_object (SSMResults): The results output of ssm_analyze.
        fontsize (int): Font size of text in the figure, in points (default = 12).
        **kwargs: Additional arguments to pass on to the plotting function.

    Returns:
        matplotlib.figure.Figure: A figure object representing the plot.
    """
    assert isinstance(ssm_object, SSMResults), "ssm_object must be an SSMResults instance"
    assert fontsize > 0, "fontsize must be a positive number"

    if ssm_object.details['contrast'] == "test":
        return ssm_plot_contrast(ssm_object, fontsize=fontsize, **kwargs)
    else:
        return ssm_plot_circle(ssm_object, fontsize=fontsize, **kwargs)


def ssm_plot_circle(ssm_object: SSMResults,
                    amax: Optional[float] = None,
                    legend_font_size: int = 12,
                    scale_font_size: int = 12,
                    lowfit: bool = True,
                    repel: bool = False,
                    angle_labels: Optional[List[str]] = None,
                    palette: Optional[str] = "Set2",
                    **kwargs
                    ):
    """
    Create a Circular Plot of SSM Results.

    Args:
        ssm_object (SSMResults): The output of ssm_analyze.
        amax (Optional[float]): A positive number corresponding to the radius of the circle.
        legend_font_size (int): Size of the text labels in the legend.
        scale_font_size (int): Size of the text labels for the amplitude and displacement scales.
        lowfit (bool): Whether profiles with low model fit (<.70) should be plotted with dashed borders.
        repel (bool): Experimental argument for plotting text labels instead of colors.
        angle_labels (Optional[List[str]]): Text labels to plot around the circle for each scale.
        palette (Optional[str]): Color palette to use for the plot.
        **kwargs: Additional arguments for matplotlib.

    Returns:
        matplotlib.figure.Figure: A figure object containing the circular plot.
    """
    df = ssm_object.results
    angles = np.round(ssm_object.details['angles']).astype(int)

    if amax is None:
        amax = np.ceil(df['a_uci'].max() * 10) / 10

    # Convert results to numbers usable by matplotlib
    df_plot = df.copy()
    df_plot['d_uci'] = np.where(df_plot['d_uci'] < df_plot['d_lci'],
                                np.deg2rad(df_plot['d_uci'] + 360),
                                np.deg2rad(df_plot['d_uci'])
                                )
    df_plot['d_lci'] = np.deg2rad(df_plot['d_lci'])
    df_plot['a_lci'] = df_plot['a_lci'] * 10 / (2 * amax)
    df_plot['a_uci'] = df_plot['a_uci'] * 10 / (2 * amax)
    df_plot['x_est'] = df_plot['x_est'] * 10 / (2 * amax)
    df_plot['y_est'] = df_plot['y_est'] * 10 / (2 * amax)

    # Remove profiles with low model fit (unless overridden)
    if not lowfit:
        df_plot = df_plot[df_plot['fit_est'] >= 0.70]
        if len(df_plot) < 1:
            raise ValueError("After removing profiles, there were none left to plot.")

    df_plot['linestyle'] = np.where(df_plot['fit_est'] >= 0.70, 'solid', 'dashed')

    fig, ax = plt.subplots(figsize=(10, 10))
    circle_base(ax, angles, amax, fontsize=scale_font_size, labels=angle_labels)

    colors = plt.cm.get_cmap(palette)(np.linspace(0, 1, len(df_plot)))

    for i, (_, row) in enumerate(df_plot.iterrows()):
        wedge = patches.Wedge((0, 0), row['a_uci'], np.rad2deg(row['d_lci']), np.rad2deg(row['d_uci']),
                              width=row['a_uci'] - row['a_lci'],
                              fc=colors[i],
                              alpha=0.4,
                              linestyle=row['linestyle']
                              )
        ax.add_patch(wedge)
        ax.plot(row['x_est'], row['y_est'], 'o', color=colors[i])

    if repel:
        for _, row in df_plot.iterrows():
            ax.annotate(row['label'], (row['x_est'], row['y_est']),
                        xytext=(-25 - row['x_est'], 0),
                        textcoords='offset points',
                        ha='right', va='center',
                        fontsize=legend_font_size,
                        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                        )
    else:
        ax.legend(df_plot['label'], loc='center left', bbox_to_anchor=(1, 0.5),
                  fontsize=legend_font_size
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
        circle = plt.Circle((0, 0), r, fill=False, color='gray')
        ax.add_artist(circle)

    # Draw lines
    for angle in np.deg2rad(angles):
        ax.plot([0, 5 * np.cos(angle)], [0, 5 * np.sin(angle)], color='gray', linewidth=0.5)

    # Add labels
    for angle, label in zip(np.deg2rad(angles), labels):
        ax.text(5.1 * np.cos(angle), 5.1 * np.sin(angle), label,
                ha='center', va='center', fontsize=fontsize
                )

    # Add amplitude labels
    ax.text(2, 0, f"{amin + (amax - amin) / 3:.2f}", ha='left', va='bottom', fontsize=fontsize)
    ax.text(4, 0, f"{amin + 2 * (amax - amin) / 3:.2f}", ha='left', va='bottom', fontsize=fontsize)

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')


def ssm_plot_contrast(ssm_object: SSMResults,
                      axislabel: str = "Difference",
                      xy: bool = True,
                      color: str = "red",
                      linesize: float = 1.25,
                      fontsize: int = 12
                      ):
    """
    Create a Difference Plot of SSM Contrast Results.

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
        'e': r'$\Delta$ Elevation',
        'x': r'$\Delta$ X-Value',
        'y': r'$\Delta$ Y-Value',
        'a': r'$\Delta$ Amplitude',
        'd': r'$\Delta$ Displacement'
        }

    pvals = ['e', 'x', 'y', 'a', 'd']

    res = ssm_object.results

    if not xy:
        res = res.drop(columns=['x_est', 'x_lci', 'x_uci', 'y_est', 'y_lci', 'y_uci'])
        plabs = {k: v for k, v in plabs.items() if k not in ['x', 'y']}
        pvals = [p for p in pvals if p not in ['x', 'y']]

    res['d_est'] = res['d_est'].astype(float)
    res['d_uci'] = np.where((res['d_uci'] < res['d_lci']) & (res['d_uci'] < 180),
                            (res['d_uci'] + 360) % 360, res['d_uci']
                            )
    res['d_lci'] = np.where((res['d_lci'] > res['d_uci']) & (res['d_lci'] > 180),
                            (res['d_lci'] - 360) % 360, res['d_lci']
                            )

    fig, axes = plt.subplots(1, len(pvals), figsize=(4 * len(pvals), 4), sharey=True)
    fig.suptitle(axislabel, fontsize=fontsize + 2)

    for ax, param in zip(axes, pvals):
        ax.axhline(y=0, color='darkgray', linewidth=linesize)
        ax.errorbar(res['label'], res[f'{param}_est'],
                    yerr=[res[f'{param}_est'] - res[f'{param}_lci'],
                          res[f'{param}_uci'] - res[f'{param}_est']],
                    fmt='o', color=color, capsize=5, capthick=linesize,
                    elinewidth=linesize, markersize=linesize * 3
                    )
        ax.set_title(plabs[param], fontsize=fontsize)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='both', labelsize=fontsize - 2)

    fig.tight_layout()
    return fig