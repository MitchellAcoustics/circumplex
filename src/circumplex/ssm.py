"""Structural Summary Method (SSM) analysis results and configuration classes.

This module provides dataclasses for representing SSM analysis results:

- SSMDetails: Configuration and parameters used in SSM analysis
- SSM: Complete SSM analysis results with estimates and confidence intervals
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from circumplex._utils import is_package_installed
from circumplex.visualization import plot_circle, plot_contrast, plot_curve

console = None
if is_package_installed("rich"):
    from rich.console import Group
    from rich.table import Table

    from circumplex import _utils

    console = getattr(_utils, "console", None)


@dataclass
class SSMDetails:
    """Details of SSM analysis configuration and parameters.

    Attributes
    ----------
    boots
        Number of bootstrap resamples used for confidence intervals.
    interval
        Confidence interval level (e.g., 0.95 for 95% CI).
    listwise
        Whether listwise deletion was used for missing data.
    angles
        Angular displacements of the circumplex scales in degrees.
    contrast
        Whether the analysis involves contrasts between groups.
    score_type
        Type of scores used in analysis ('mean' or 'correlation').Pterm
    """

    boots: int
    interval: float
    listwise: bool
    angles: list[float]
    contrast: bool
    score_type: str

    @classmethod
    def from_dict(cls, data: dict) -> SSMDetails:
        """Create SSMDetails instance from dictionary.

        Parameters
        ----------
        data
            Dictionary containing SSM analysis parameters with keys:
            boots, interval, listwise, angles, contrast, score_type.

        Returns
        -------
        New SSMDetails instance populated from dictionary data.
        """
        return cls(
            boots=data["boots"],
            interval=data["interval"],
            listwise=data["listwise"],
            angles=data["angles"],
            contrast=data["contrast"],
            score_type=data["score_type"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert SSMDetails instance to dictionary.

        Returns
        -------
        Dictionary containing all SSMDetails attributes.
        """
        return self.__dict__.copy()

    def __rich_repr__(self) -> Iterable[str]:
        """Generate rich console representation of SSM analysis details."""
        yield f"Statistical Basis:   {self.score_type.capitalize()} Scores"
        yield f"Bootstrap Resamples: {self.boots}"
        yield f"Confidence Level:    {self.interval}"
        yield f"Listwise Deletion:   {self.listwise}"
        yield f"Scale Displacements: {self.angles}"


@dataclass
class SSM:
    """Results from a Structural Summary Method (SSM) analysis.

    Attributes
    ----------
    results
        DataFrame containing SSM parameter estimates and confidence intervals.
    scores
        DataFrame containing the circumplex scale scores used in the analysis.
    details
        Configuration and parameters used in the SSM analysis.
    type
        Type of SSM analysis performed (e.g., 'profile', 'contrast').
    """

    results: pd.DataFrame
    scores: pd.DataFrame
    details: SSMDetails
    type: str

    @classmethod
    def from_dict(cls, data: dict) -> SSM:
        """Create SSM instance from dictionary.

        Parameters
        ----------
        data
            Dictionary containing SSM analysis results with keys:
            results, scores, details, type.

        Returns
        -------
        New SSM instance populated from dictionary data.
        """
        return cls(
            results=data["results"],
            scores=data["scores"],
            details=SSMDetails.from_dict(data["details"]),
            type=data["type"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert SSM instance to dictionary.

        Returns
        -------
        Dictionary containing all SSM attributes with details converted to dict format.
        """
        d = self.__dict__.copy()
        d["details"] = self.details.to_dict()
        return d

    def summary(self, *, rich_print: bool = True) -> None:
        """Print a formatted summary of SSM analysis results.

        Parameters
        ----------
        rich_print
            Whether to use rich console formatting for output, by default True.
            If False or rich is not installed, falls back to standard printing.
        """
        summ_secs = []  # Initialize summ_secs here
        if is_package_installed("rich") and rich_print:
            summ_secs = [*self.details.__rich_repr__()]
            summ_secs.append("\n")

            for _, row in self.results.iterrows():
                tbl = Table(title=f"Profile[{row.Label}]")
                tbl.add_column("")
                tbl.add_column("Estimate")
                tbl.add_column("Lower CI")
                tbl.add_column("Upper CI")

                for label, val in zip(
                    ["Elevation", "X-Value", "Y-Value", "Amplitude", "Displacement"],
                    ["e", "x", "y", "a", "d"],
                    strict=False,
                ):
                    tbl.add_row(
                        label,
                        str(round(row[f"{val}_est"], 3)),
                        str(round(row[f"{val}_lci"], 3)),
                        str(round(row[f"{val}_uci"], 3)),
                    )
                tbl.add_row("Model Fit", str(round(row.fit_est, 3)))

                summ_secs.append(tbl)

            summ_group = Group(*summ_secs)  # This will now work correctly
            if console is not None:
                console.print(summ_group)
        else:
            # Non-rich version of the summary
            print(f"Statistical Basis: {self.details.score_type.capitalize()} Scores")  # noqa: T201
            print(f"Bootstrap Resamples: {self.details.boots}")  # noqa: T201
            print(f"Confidence Level: {self.details.interval}")  # noqa: T201
            print(f"Listwise Deletion: {self.details.listwise}")  # noqa: T201
            print(f"Scale Displacements: {self.details.angles}")  # noqa: T201
            print("\nResults:")  # noqa: T201
            for _, row in self.results.iterrows():
                print(f"Profile[{row.Label}]:")  # noqa: T201
                print(  # noqa: T201
                    f"  Elevation: {round(row.e_est, 3)}, "
                    f"Lower CI: {round(row.e_lci, 3)}, "
                    f"Upper CI: {round(row.e_uci, 3)}"
                )
                print(  # noqa: T201
                    f"  X-Value: {round(row.x_est, 3)}, "
                    f"Lower CI: {round(row.x_lci, 3)}, "
                    f"Upper CI: {round(row.x_uci, 3)}"
                )
                print(  # noqa: T201
                    f"  Y-Value: {round(row.y_est, 3)}, "
                    f"Lower CI: {round(row.y_lci, 3)}, "
                    f"Upper CI: {round(row.y_uci, 3)}"
                )
                print(  # noqa: T201
                    f"  Amplitude: {round(row.a_est, 3)}, "
                    f"Lower CI: {round(row.a_lci, 3)}, "
                    f"Upper CI: {round(row.a_uci, 3)}"
                )
                print(f"  Displacement: {round(row.d_est, 3)}")  # noqa: T201
                print(f"  Model Fit: {round(row.fit_est, 3)}\n")  # noqa: T201

    def plot_circle(self, **kwargs: Any) -> Figure:
        """Generate a circular SSM plot for the analysis results.

        Convenience method that passes SSM data to the plot_circle() function.
        Automatically plots all profiles in the results.

        Parameters
        ----------
        **kwargs
            Additional plotting options passed to plot_circle(). See plot_circle()
            documentation for available options (e.g., amax, angle_labels, colors,
            fontsize, drop_lowfit, figsize, title).

        Returns
        -------
        Matplotlib Figure object.

        Examples
        --------
        >>> results = ssm_analyze(data, scales=list(range(8)))
        >>> fig = results.plot_circle()
        >>> fig.savefig('profile.png')

        >>> fig = results.plot_circle(colors="husl", fontsize=14)

        See Also
        --------
        circumplex.visualization.plot_circle : Full function documentation
        """
        return plot_circle(
            results_df=self.results,
            angles=self.details.angles,
            **kwargs,
        )

    def plot_curve(self, **kwargs: Any) -> Figure:
        """Generate SSM curve plots for the analysis results.

        Convenience method that passes SSM data to the plot_curve() function.
        Creates faceted plots showing fitted curves overlaid on observed scores.

        Parameters
        ----------
        **kwargs
            Additional plotting options passed to plot_curve(). See plot_curve()
            documentation for available options (e.g., angle_labels, colors,
            base_size, drop_lowfit, figsize).

        Returns
        -------
        Matplotlib Figure object.

        Examples
        --------
        >>> results = ssm_analyze(data, scales=list(range(8)))
        >>> fig = results.plot_curve()
        >>> fig.savefig('curves.png')

        >>> fig = results.plot_curve(angle_labels=['PA', 'BC', 'DE', 'FG',
        ...                                         'HI', 'JK', 'LM', 'NO'])

        See Also
        --------
        circumplex.visualization.plot_curve : Full function documentation
        """
        return plot_curve(
            results_df=self.results,
            scores_df=self.scores,
            angles=self.details.angles,
            **kwargs,
        )

    def plot_contrast(self, **kwargs: Any) -> Figure:
        """Generate SSM parameter contrast plots for the analysis results.

        Convenience method that passes SSM data to the plot_contrast() function.
        Only available for contrast analyses (when contrast=True in ssm_analyze).

        Parameters
        ----------
        **kwargs
            Additional plotting options passed to plot_contrast(). See
            plot_contrast() documentation for available options (e.g., drop_xy,
            sig_color, ns_color, linewidth, fontsize, figsize).

        Returns
        -------
        Matplotlib Figure object.

        Raises
        ------
        ValueError
            If this SSM object does not contain contrast results.

        Examples
        --------
        >>> results = ssm_analyze(data, scales=list(range(8)),
        ...                       grouping='condition', contrast=True)
        >>> fig = results.plot_contrast()
        >>> fig.savefig('contrasts.png')

        >>> fig = results.plot_contrast(drop_xy=True)

        See Also
        --------
        circumplex.visualization.plot_contrast : Full function documentation
        """
        if not self.details.contrast:
            msg = (
                "plot_contrast() requires contrast results. "
                "Run ssm_analyze() with contrast=True."
            )
            raise ValueError(msg)

        return plot_contrast(
            results_df=self.results,
            **kwargs,
        )
