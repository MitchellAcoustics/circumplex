import pandas as pd
from typing import Dict, Any


class SSMResults:
    def __init__(
        self,
        results: pd.DataFrame,
        scores: pd.DataFrame,
        details: Dict[str, Any],
        call: str,
    ):
        self.results = results
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
                f"Scale Displacements:   {self.details['angles'].tolist()}\n",
            ]
        )

        output.extend(str(self).split("\n")[2:])  # Add the formatted results
        return "\n".join(output)
