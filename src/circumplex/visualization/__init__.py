"""Circumplex visualization module.

This module provides functions for visualizing SSM analysis results using
matplotlib. Three main plot types are available:

- plot_circle: Circular plot showing amplitude and displacement
- plot_curve: Faceted plot showing fitted cosine curves with observed scores
- plot_contrast: Faceted plots for parameter contrasts between groups
"""

from circumplex.visualization.plots import plot_circle, plot_contrast, plot_curve

__all__ = ["plot_circle", "plot_contrast", "plot_curve"]
