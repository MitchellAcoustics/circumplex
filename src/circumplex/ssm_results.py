from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import circumplex.visualization as vis


class SSMResults:
    def __init__(
        self,
        results: pd.DataFrame,
        scales: List[str],
        scores: pd.DataFrame,
        details: Dict[str, Any],
        call: str,
    ):
        self.results = results
        self.scales = scales
        self.scores = scores
        self.details = details
        self.call = call

    def __str__(self):
        output = [f"Call:\n{self.call}\n"]
        for _, row in self.results.iterrows():
            output.append(f"\n{self.details['results_type']} [{row['label']}]:")
            params = [
                "Elevation",
                "X-Value",
                "Y-Value",
                "Amplitude",
                "Displacement",
                "Model Fit",
            ]
            output.append(f"{'Parameter     ':<10} {'Estimate   ':>8} [LCI, UCI]")
            param_abbrev = [f"{p.lower()[0]}" for p in params]
            param_abbrev.pop(-1)
            param_abbrev.append("fit")
            estimates = [row[f"{p}_est"] for p in param_abbrev]
            lower_ci = [row[f"{p}_lci"] for p in param_abbrev]
            upper_ci = [row[f"{p}_uci"] for p in param_abbrev]

            max_len = max(len(p) for p in params)
            for param, est, lci, uci in zip(params, estimates, lower_ci, upper_ci):
                output.append(
                    f"{param:<{max_len}} {est:>8.3f} [{lci:>8.3f}, {uci:>8.3f}]"
                )

        return "\n".join(output)

    def summary(self):
        output = [f"Call:\n{self.call}\n"]
        output.extend(
            [
                f"Statistical Basis:     {self.details['score_type']} Scores",
                f"Bootstrap Resamples:   {self.details['boots']}",
                f"Confidence Level:      {self.details['interval']}",
                f"Listwise Deletion:     {self.details['listwise']}",
                f"Scale Displacements:   {self.details['angles']}\n",
            ]
        )

        output.extend(str(self).split("\n")[2:])  # Add the formatted results
        return "\n".join(output)

    def table(self):
        table = pd.DataFrame(
            columns=[
                "Profile",
                "Elevation",
                "X-Value",
                "Y-Value",
                "Amplitude",
                "Displacement",
                "Fit",
            ],
            index=self.results.index,
        )
        for i, (_, row) in enumerate(self.results.iterrows()):
            table.loc[i] = [
                row["label"],
                row["e_est"],
                row["x_est"],
                row["y_est"],
                row["a_est"],
                row["d_est"],
                row["fit_est"],
            ]
        return table

    def plot(self, **kwargs) -> Figure:
        """
        Create a figure from SSM results.

        This is a convenience wrapper for ssm_plot().

        Parameters
        ----------
        **kwargs
            Additional arguments to pass to ssm_plot().

        Returns
        -------
        matplotlib.figure.Figure
            A figure object representing the plot.
        """
        return vis.ssm_plot(self, **kwargs)

    def profile_plot(self, **kwargs) -> Tuple[Figure, List[Axes]]:
        """
        Create profile plots from SSM results.

        Creates one profile plot for each component in the results.

        Parameters
        ----------
        **kwargs
            Additional arguments to pass to ssm_profile_plot().

        Returns
        -------
        Tuple[matplotlib.figure.Figure, List[matplotlib.axes.Axes]]
            A tuple containing the figure and axes objects.
        """
        results = self.results
        scores = self.scores
        details = self.details

        n_components = results.shape[0]
        fig, axes = plt.subplots(n_components, 1, figsize=(6, 4 * n_components))
        if n_components == 1:
            axes = [axes]

        for i, (ax, (_, row)) in enumerate(zip(axes, results.iterrows())):
            fig, ax = vis.ssm_profile_plot(
                scores=scores.iloc[i][self.scales],
                angles=details["angles"],
                amplitude=row["a_est"],
                displacement=row["d_est"],
                elevation=row["e_est"],
                r2=row["fit_est"],
                title=f"{results.iloc[i]['label']} Profile",
                ax=ax,
                **kwargs,
            )

        plt.tight_layout()

        return fig, axes
