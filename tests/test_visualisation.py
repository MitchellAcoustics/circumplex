import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.collections import PathCollection
from matplotlib.patches import Circle, Wedge

from circumplex import SSMResults, visualization

SCALES = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]


@pytest.fixture
def mock_ssm_results():
    results = pd.DataFrame(
            {
                "label"  : ["Group A", "Group B"],
                "e_est"  : [0.5, 0.7],
                "x_est"  : [1.0, -0.5],
                "y_est"  : [0.5, 1.0],
                "a_est"  : [1.12, 1.12],
                "d_est"  : [26.57, 116.57],
                "fit_est": [1.0, 1.0],
                "e_lci"  : [0.3, 0.5],
                "x_lci"  : [0.8, -0.7],
                "y_lci"  : [0.3, 0.8],
                "a_lci"  : [0.9, 0.9],
                "d_lci"  : [20.0, 110.0],
                "fit_lci": [0.7, 0.8],
                "e_uci"  : [0.7, 0.9],
                "x_uci"  : [1.2, -0.3],
                "y_uci"  : [0.7, 1.2],
                "a_uci"  : [1.3, 1.3],
                "d_uci"  : [33.0, 123.0],
                "fit_uci": [0.9, 1.0],
                }
            )
    scores = pd.DataFrame(
            {
                "V1"   : [1.50, 0.20],
                "V2"   : [1.56, 1.05],
                "V3"   : [1.00, 1.70],
                "V4"   : [0.15, 1.76],
                "V5"   : [-0.50, 1.20],
                "V6"   : [-0.56, 0.35],
                "V7"   : [0, -0.30],
                "V8"   : [0.85, -0.36],
                "label": ["Group A", "Group B"],
                }
            )
    details = {
        "angles"    : (0, 45, 90, 135, 180, 225, 270, 315),
        "score_type": "Mean",
        "results_type": "Profile",
        "contrast"  : "none",
        }
    return SSMResults(
            results=results, scores=scores, details=details, call="mock_call", scales=SCALES
            )


def test_ssm_plot(mock_ssm_results):
    fig = visualization.ssm_plot(mock_ssm_results)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1  # Assuming ssm_plot creates a single plot
    ax = fig.axes[0]

    # Check if the plot contains the expected elements
    assert ax.get_legend() is not None, "Legend is missing from the plot"
    assert len(ax.get_lines()) > 0, "No lines found in the plot"
    assert (
            len([c for c in ax.get_children() if isinstance(c, PathCollection)]) > 0
    ), "No scatter points found in the plot"

    plt.close(fig)


def test_ssm_plot_circle(mock_ssm_results):
    fig = visualization.ssm_plot_circle(mock_ssm_results)
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    children = ax.get_children()

    # Check axis limits
    assert ax.get_xlim() == (-5.5, 5.5)
    assert ax.get_ylim() == (-5.5, 5.5)

    # Check if circular grid is present (5 concentric circles)
    circles = [child for child in children if isinstance(child, Circle)]
    assert (
            len(circles) == 5
    ), f"Expected 5 circular grid lines, but found {len(circles)}"

    # Check if radial lines are present (8 lines for octants)
    radial_lines = [child for child in children if isinstance(child, plt.Line2D)]
    assert (
            len(radial_lines) >= 8
    ), f"Expected at least 8 radial lines, but found {len(radial_lines)}"

    # Check if angle labels are present
    angle_labels = [
        child
        for child in children
        if isinstance(child, plt.Text) and "°" in child.get_text()
        ]
    assert (
            len(angle_labels) == 8
    ), f"Expected 8 angle labels, but found {len(angle_labels)}"

    # Check if wedges (confidence regions) are plotted
    wedges = [child for child in children if isinstance(child, Wedge)]
    assert len(wedges) == len(
            mock_ssm_results.results
            ), f"Expected {len(mock_ssm_results.results)} wedges, but found {len(wedges)}"

    # Check if scatter points are plotted
    scatter_points = [child for child in children if isinstance(child, PathCollection)]
    assert (
            len(scatter_points) == 1
    ), f"Expected 1 scatter collection, but found {len(scatter_points)}"
    assert (
            len(scatter_points[0].get_offsets()) == len(mock_ssm_results.results)
    ), f"Expected {len(mock_ssm_results.results)} scatter points, but found {len(scatter_points[0].get_offsets())}"

    # Check if legend is present
    assert ax.get_legend() is not None, "Legend is missing from the plot"

    plt.close(fig)


def test_ssm_plot_contrast(mock_ssm_results):
    fig = visualization.ssm_plot_circle(mock_ssm_results)
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    children = ax.get_children()

    # Check axis limits
    assert ax.get_xlim() == (-5.5, 5.5)
    assert ax.get_ylim() == (-5.5, 5.5)

    # Check if circular grid is present (5 concentric circles)
    circles = [child for child in children if isinstance(child, Circle)]
    assert (
            len(circles) == 5
    ), f"Expected 5 circular grid lines, but found {len(circles)}"

    # Check if radial lines are present (8 lines for octants)
    radial_lines = [child for child in children if isinstance(child, plt.Line2D)]
    assert (
            len(radial_lines) >= 8
    ), f"Expected at least 8 radial lines, but found {len(radial_lines)}"

    # Check if angle labels are present
    angle_labels = [
        child
        for child in children
        if isinstance(child, plt.Text) and "°" in child.get_text()
        ]
    assert (
            len(angle_labels) == 8
    ), f"Expected 8 angle labels, but found {len(angle_labels)}"

    # Check if wedges (confidence regions) are plotted
    wedges = [child for child in children if isinstance(child, Wedge)]
    assert len(wedges) == len(
            mock_ssm_results.results
            ), f"Expected {len(mock_ssm_results.results)} wedges, but found {len(wedges)}"

    # Check if scatter points are plotted
    scatter_points = [child for child in children if isinstance(child, PathCollection)]
    assert (
            len(scatter_points) == 1
    ), f"Expected 1 scatter collection, but found {len(scatter_points)}"
    assert (
            len(scatter_points[0].get_offsets()) == len(mock_ssm_results.results)
    ), f"Expected {len(mock_ssm_results.results)} scatter points, but found {len(scatter_points[0].get_offsets())}"

    # Check if legend is present
    assert ax.get_legend() is not None, "Legend is missing from the plot"

    plt.close(fig)  # Close the figure to free up memory


def test_ssm_plot_profile(mock_ssm_results):
    fig, axes = mock_ssm_results.profile_plot()
    assert isinstance(fig, plt.Figure)
    assert len(axes) == len(mock_ssm_results.results)

    for ax in axes:
        # Check if the main line plot exists
        lines = [child for child in ax.get_children() if isinstance(child, plt.Line2D)]
        assert any(
                len(line.get_xdata()) > 2 for line in lines
                ), "Main plot line is missing"

        # Check for x-axis labels (angles)
        assert len(ax.get_xticklabels()) == 8, "Incorrect number of x-axis labels"

        # Check if y-axis label is present
        assert ax.get_ylabel() != "", "Y-axis label is missing"

        # Check if title is present
        assert ax.get_title() != "", "Plot title is missing"

    plt.close(fig)


def test_invalid_input():
    with pytest.raises((AssertionError, TypeError)):
        visualization.ssm_plot("not an SSMResults object")


# Run the tests
if __name__ == "__main__":
    pytest.main()
