# %%
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from dataclasses import dataclass
import json
from importlib.resources import files
import matplotlib.pyplot as plt
import numpy as np
from circumplex import SSMResults, ssm_analyse

INSTRUMENT_JSONS = {
    "CSIP": str(files("circumplex.instruments").joinpath("CSIP.json")),
    "SSQP-eng": str(files("circumplex.instruments").joinpath("SSQP-eng.json")),
    "SATP-eng": str(files("circumplex.instruments").joinpath("SATP-eng.json")),
}


def from_dict(inst_dict: dict) -> Instrument:
    """
    Compose an Instrument object from a dictionary.

    Typically this would be used to load an instrument from one of our built in JSON files.
    Args:
        inst_dict: A dictionary containing the instrument's details, scales, anchors, and items.

    Returns:
        Instrument: An Instrument object.
    """
    scales = Scales(
        abbrev=list(inst_dict["scales"].keys()),
        label=[scale["label"] for scale in inst_dict["scales"].values()],
        angle=[scale["angle"] for scale in inst_dict["scales"].values()],
    )
    items = None
    anchors = Anchors(
        value=[int(key) for key in inst_dict["anchors"].keys()],
        label=list(inst_dict["anchors"].values()),
    )
    details = InstrumentDetails(**inst_dict["details"])
    return Instrument(scales, anchors, details, items)


def load_instrument(instrument: str) -> Instrument:
    """
    Load an instrument from one of our built-in JSON files.

    Args:
        instrument: The name of the instrument to load. Must be one of the following:
            - CSIP

    Returns:
        Instrument: An Instrument object.
    """
    with open(INSTRUMENT_JSONS[instrument], "r") as f:
        instrument = json.load(f)

    return from_dict(instrument)


@dataclass
class Anchors:
    value: list[int]
    label: list[str]

    def __post_init__(self):
        assert len(self.value) == len(self.label)

    def __repr__(self):
        return f"Anchors({self.value}, {self.label})"

    def __str__(self):
        return "\n".join(
            [f"{value}. {label}" for value, label in zip(self.value, self.label)]
        )


@dataclass
class Items:
    number: list[int]
    text: list[str]

    def __post_init__(self):
        assert len(self.number) == len(self.text)
        assert len(self.text) == len(set(self.text)), "Items must be unique"

    def __str__(self):
        return "\n".join(
            [f"{number}. {text}" for number, text in zip(self.number, self.text)]
        )

    def get_items(self, numbers: list[int]) -> Items:
        return Items(numbers, [self.text[number] for number in numbers])


@dataclass
class Scales:
    abbrev: list[str]
    label: list[str]
    angle: list[float]
    items: list[int] | None = None
    data: pd.DataFrame | None = None

    def __post_init__(self):
        assert len(self.abbrev) == len(self.angle)
        assert len(self.abbrev) == len(set(self.abbrev)), "Abbreviations must be unique"
        assert (
            max(self.angle) <= 360 and min(self.angle) >= 0
        ), "Angles must be between 0 and 360"

    def __str__(self, items: bool = False):
        # TODO: Add print method for items = True
        if items is False:
            return "\n".join(
                [
                    f"{abbrev}: {label} ({angle}°)"
                    for abbrev, label, angle in zip(self.abbrev, self.label, self.angle)
                ]
            )


@dataclass
class InstrumentDetails:
    name: str
    abbrev: str
    items: int | None = None
    scales: int | None = None
    prefix: str | None = None
    suffix: str | None = None
    status: str | None = None
    construct: str | None = None
    reference: str | None = None
    url: str | None = None

    def __str__(self):
        return (
            f"{self.abbrev}: {self.name}\n"
            f"{self.items} Items, {self.scales} Scales\n"
            f"{self.reference}\n"
            f"<{self.url}>"
        )


@dataclass
class Instrument:
    """
    A class for representing circumplex instruments.

    Attributes:
        scales: Scales
        anchors: Anchors
        details: InstrumentDetails
        items: Items | None = None
        _data: pd.DataFrame | None = None
    """

    scales: Scales
    anchors: Anchors
    details: InstrumentDetails
    items: Items | None = None
    _data: pd.DataFrame | None = None

    def __repr__(self):
        return (
            f"{self.details.abbrev}: {self.details.name}\n"
            f"{self.details.items} Items, {self.details.scales} Scales\n"
            f"{self.details.reference}\n"
            f"<{self.details.url}>"
        )

    @property
    def data(self):
        if self._data is None:
            raise UserWarning(
                "No data has been loaded for this instrument. Use attach_data() to load data."
            )
        else:
            return self._data

    def summary(self):
        print(self.details)
        print(
            f"\nThe {self.details.abbrev} contains {self.details.scales} circumplex scales."
        )
        print(self.scales)
        print(
            f"\nThe {self.details.abbrev} is rated using the following {len(self.anchors.value)}-point scale."
        )
        print(self.anchors)
        print(
            f"\nThe {self.details.abbrev} contains {self.details.items} items ({self.details.status})."
        )
        print(self.items)

    def attach_data(self, data: pd.DataFrame) -> Instrument:
        # check scales
        assert set(self.scales.abbrev).issubset(data.columns), (
            f"Data is missing scales. "
            f"Missing scales: {set(self.scales.abbrev) - set(data.columns)}"
        )
        self._data = data
        return self

    def ssm_analyse(
        self, measures: list[str] = None, grouping: list[str] = None
    ) -> SSMResults:
        return ssm_analyse(
            self.data,
            self.scales.abbrev,
            measures=measures,
            grouping=grouping,
            angles=tuple(self.scales.angle),
        )

    def demo_plot(self):
        # alabel = self.scales.label
        # angles = self.scales.angle
        degree_sign = "\N{DEGREE SIGN}"

        # Create plot ---------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))

        ax.plot()
        ax.tick_params(axis="both", pad=10)
        ax.set_xticks(
            np.radians(self.scales.angle),
            labels=self.scales.label,
            fontsize=12,
        )

        ax.set_yticks([])
        ax.grid(True)
        for i, angle in enumerate(self.scales.angle):
            ax.text(
                np.radians(angle),
                0.4,
                f"{angle}{degree_sign}",
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
            )
            ax.text(
                np.radians(angle),
                0.75,
                self.scales.abbrev[i],
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
            )
        plt.show()
