# %%
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from circumplex import SSMResults, ssm_analyse

INSTRUMENT_JSONS = {
    "CSIP": str(files("circumplex.instruments").joinpath("CSIP.json")),
    "IIPSC": str(files("circumplex.instruments").joinpath("IIPSC.json")),
    "SSQP-eng": str(files("circumplex.instruments").joinpath("SSQP-eng.json")),
    "SATP-eng": str(files("circumplex.instruments").joinpath("SATP-eng.json")),
}


def instruments() -> None:
    """
    Print a list of the instruments included in the circumplex package.

    Args:
        None

    Returns:
        None
    """
    ins = {name: load_instrument(name) for name in INSTRUMENT_JSONS.keys()}
    print(f"The circumplex package currently includes {len(ins)} instruments:")
    i = 1
    for name, inst in ins.items():
        print(f"{i}. {name}: {inst.details.name} ({inst.details.abbrev})")
        i += 1

    return None


def from_dict(inst_dict: dict) -> Instrument:
    """
    Compose an Instrument object from a dictionary.

    Typically this would be used to load an instrument from one of our built in JSON files.
    Args:
        inst_dict: A dictionary containing the instrument's details, scales, anchors, and items.

    Returns:
        Instrument: An Instrument object.
    """
    items_exist = sum(
        ["inst_items" in scale.keys() for scale in inst_dict["scales"].values()]
    )
    scales = Scales(
        abbrev=list(inst_dict["scales"].keys()),
        label=[scale["label"] for scale in inst_dict["scales"].values()],
        angle=[scale["angle"] for scale in inst_dict["scales"].values()],
        inst_items=[
            inst_dict["scales"][scale]["inst_items"]
            for scale in inst_dict["scales"].keys()
        ]
        if items_exist
        else None,
    )
    anchors = Anchors(
        value=[int(key) for key in inst_dict["anchors"].keys()],
        label=list(inst_dict["anchors"].values()),
    )
    norms = (
        Norms(
            table=pd.DataFrame.from_dict(inst_dict["norms"]),
            src=pd.DataFrame.from_dict(inst_dict["norms_src"]),
        )
        if "norms" in inst_dict
        else None
    )
    details = InstrumentDetails(**inst_dict["details"])
    return Instrument(scales, anchors, details, norms)


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

    def show(self):
        return print(self)


@dataclass
class Items:
    data: dict

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self.data[key] = value

    def __delitem__(self, key: Any) -> None:
        del self.data[key]

    def keys(self) -> list:
        return list(self.data.keys())

    def values(self) -> list:
        return list(self.data.values())

    def inst_items(self) -> list:
        return list(self.data.items())

    def __str__(self):
        return "\n".join([f"{number}. {text}" for number, text in self.data.items()])

    def show(self, n=10):
        n = len(self.data) if n is None or n > len(self.data) else n
        p = "\n".join(
            [f"{number}. {text}" for number, text in list(self.data.items())[:n]]
        )
        if n < len(self.data):
            p += f"\n\n...and {len(self.data) - n} more items."

        return print(p)


@dataclass
class Scales:
    abbrev: list[str]
    label: list[str]
    angle: list[float]
    inst_items: list[dict] | None = None

    def __post_init__(self):
        assert len(self.abbrev) == len(self.angle)
        assert len(self.abbrev) == len(set(self.abbrev)), "Abbreviations must be unique"
        assert (
            max(self.angle) <= 360 and min(self.angle) >= 0
        ), "Angles must be between 0 and 360"

    def __str__(self):
        return "\n".join(
            [
                f"{abbrev}: {label} ({angle}°)"
                for abbrev, label, angle in zip(self.abbrev, self.label, self.angle)
            ]
        )

    def show(self, inst_items: bool = True):
        if inst_items is False:
            return print(self)
        else:
            p = []
            for i, abbrev in enumerate(self.abbrev):
                p.append(f"{abbrev}: {self.label[i]} ({self.angle[i]}°)")
                p.append(
                    "\n".join(
                        [f"\t{key}: {val}" for key, val in self.inst_items[i].items()]
                    )
                )
            return print("\n".join(p))


@dataclass
class Norms:
    table: pd.DataFrame
    src: pd.DataFrame

    def get_sample(self, sample: int) -> pd.DataFrame:
        return self.table.query("sample == @sample")

    def show(self):
        return print(self.src)


@dataclass
class InstrumentDetails:
    name: str
    abbrev: str
    inst_items: int | None = None
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
            f"{self.inst_items} Items, {self.scales} Scales\n"
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
        inst_items: Items | None = None
        _data: pd.DataFrame | None = None
    """

    scales: Scales
    anchors: Anchors
    details: InstrumentDetails
    norms: Norms | None = None
    _data: pd.DataFrame | None = None

    def __repr__(self):
        return (
            (
                f"{self.details.abbrev}: {self.details.name}\n"
                f"{self.details.inst_items} Items, {self.details.scales} Scales\n"
                f"{self.details.reference}\n"
                f"<{self.details.url}>"
            )
            if self.norms is None
            else (
                f"{self.details.abbrev}: {self.details.name}\n"
                f"{self.details.inst_items} Items, {self.details.scales} Scales, {len(self.norms.src)} normative data sets\n"
                f"{self.details.reference}\n"
                f"<{self.details.url}>"
            )
        )

    @property
    def data(self):
        if self._data is None:
            raise UserWarning(
                "No data has been loaded for this instrument. Use attach_data() to load data."
            )
        else:
            return self._data

    @property
    def inst_items(self):
        if self.scales.inst_items is None:
            raise UserWarning("No items have been defined for this instrument.")
        else:
            item_dict = {}
            for val in self.scales.inst_items:
                for key, value in val.items():
                    item_dict[int(key)] = value
            item_dict = {k: v for k, v in sorted(item_dict.items())}

            return Items(item_dict)

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
            f"\nThe {self.details.abbrev} contains {self.details.inst_items} items ({self.details.status})."
        )
        try:
            print(self.inst_items)
        except UserWarning:
            print("\nNo items have been defined for this instrument.")
        try:
            print(self.data)
        except UserWarning:
            print(
                "\nNo data has been loaded for this instrument. Use attach_data() to load data."
            )

    def attach_data(self, data: pd.DataFrame, scales: list | dict = None) -> Instrument:
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
