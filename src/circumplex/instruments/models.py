"""Instrument data models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from circumplex._utils import is_package_installed

console = None  # Will be set if rich is available
if is_package_installed("rich"):
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree

    from circumplex import _utils

    console = getattr(_utils, "console", None)


@dataclass(frozen=True)
class InstrumentScale:
    """Single scale within an instrument."""

    abbrev: str
    angle: float  # degrees
    items: tuple[int, ...]  # immutable
    label: str


@dataclass(frozen=True)
class ResponseAnchor:
    """Response option for an instrument."""

    value: int
    label: str


@dataclass(frozen=True)
class ResponseItem:
    """Single item within an instrument."""

    item_id: int
    text: str


@dataclass(frozen=True)
class NormativeSample:
    """Normative sample metadata and statistics."""

    sample_id: int
    size: int
    population: str
    reference: str
    url: str
    statistics: pd.DataFrame  # columns: scale, mean, sd, etc.


@dataclass(frozen=True)
class Instrument:
    """Circumplex instrument definition."""

    # Metadata
    name: str
    abbrev: str
    construct: str
    reference: str
    url: str
    status: str  # "open-access" or "limited"

    # Structure
    scales: tuple[InstrumentScale, ...]
    anchors: tuple[ResponseAnchor, ...]
    items: tuple[ResponseItem, ...] | None = None  # Full item text (if available)

    # Normative data
    norms: tuple[NormativeSample, ...] = ()

    # Formatting
    prefix: str = ""
    suffix: str = ""

    @property
    def n_scales(self) -> int:
        """Number of scales in the instrument."""
        return len(self.scales)

    @property
    def n_items(self) -> int:
        """Number of items in the instrument."""
        return len(self.items) if self.items else 0

    @property
    def scale_labels(self) -> list[str]:
        """Get names of all scales."""
        return [scale.label for scale in self.scales]

    @property
    def scale_abbrevs(self) -> list[str]:
        """Get abbreviations of all scales."""
        return [scale.abbrev for scale in self.scales]

    def get_angles(self) -> list[float]:
        """Get angular positions for all scales."""
        return [scale.angle for scale in self.scales]

    def get_scale(self, abbrev: str) -> InstrumentScale:
        """Get scale by abbreviation."""
        for scale in self.scales:
            if scale.abbrev == abbrev:
                return scale
        msg = (
            f"Scale with abbreviation '{abbrev}' not found in instrument "
            f"'{self.abbrev}'."
        )
        raise ValueError(msg)

    def get_item(self, item_id: int) -> ResponseItem:
        """Get item by item ID."""
        if self.items is None:
            msg = f"Instrument '{self.abbrev}' does not have item text available."
            raise ValueError(msg)
        for item in self.items:
            if item.item_id == item_id:
                return item
        msg = f"Item with ID '{item_id}' not found in instrument '{self.abbrev}'."
        raise ValueError(msg)

    def __repr__(self) -> str:
        """Return a human-readable multi-line summary of the instrument."""
        lines = [
            f"{self.abbrev}: {self.name}",
            (
                f"{self.n_items} items, {self.n_scales} scales, "
                f"{len(self.norms)} normative data sets"
            ),
            f"{self.reference}",
            f"< {self.url} >",
        ]
        return "\n".join(lines)

    def __rich_repr__(self) -> Iterable[str]:
        """Yield lines for rich-rendered representation."""
        yield f"{self.abbrev}: {self.name}"
        yield (
            f"{self.n_items} items, {self.n_scales} scales, "
            f"{len(self.norms)} normative data sets"
        )
        yield f"{self.reference}"
        yield f"< {self.url} >"

    def info(
        self,
        *,
        scales: bool = True,
        anchors: bool = True,
        items: bool = False,
        norms: bool = True,
        rich_print: bool = True,
    ) -> None:
        """Print instrument information."""
        if is_package_installed("rich") and rich_print:
            info_sections = [
                *self.__rich_repr__(),
            ]
            if scales:
                info_sections.append("\n")
                info_sections.append(
                    self.info_scales(items=items, rich_print=rich_print)
                )
            if anchors:
                info_sections.append("\n")
                info_sections.append(self.info_anchors(rich_print=rich_print))
            if norms:
                info_sections.append("\n")
                info_sections.append(self.info_norms(rich_print=rich_print))
            info_group = Group(*info_sections)
            if console is not None:
                console.print(info_group)
        else:
            print(self)  # noqa: T201
            print()  # noqa: T201
            if scales:
                print(self.info_scales(items=items, rich_print=rich_print))  # noqa: T201
                print()  # noqa: T201
            if anchors:
                print(self.info_anchors(rich_print=rich_print))  # noqa: T201
                print()  # noqa: T201
            if norms:
                print(self.info_norms(rich_print=rich_print))  # noqa: T201
                print()  # noqa: T201

    def info_scales(
        self, *, items: bool = False, rich_print: bool = True
    ) -> str | Tree:
        """Return information about instrument scales, optionally with items."""
        if is_package_installed("rich") and rich_print:
            tree = Tree(
                f"[cyan]The {self.abbrev} contains {self.n_scales} scales:",
                style="bold",
                guide_style="dim",
            )
            for scale in self.scales:
                scale_branch = tree.add(
                    f"{scale.abbrev} ({scale.angle}°): {scale.label}"
                )
                if items and self.items is not None:
                    for item_id in scale.items:
                        scale_branch.add(
                            f"[dim]{item_id}. {self.items[item_id - 1].text}"
                        )
            return tree

        text = [f"The {self.abbrev} contains {self.n_scales} scales:"]
        for scale in self.scales:
            text.append(f"  {scale.abbrev}: {scale.label} ({scale.angle}°)")
            if items and self.items is not None:
                for item_id in scale.items:
                    text.append(f"    {item_id}. {self.items[item_id - 1].text}")
        return "\n".join(text)

    def info_anchors(self, *, rich_print: bool = True) -> str | Text:
        """Return the response anchors for the instrument."""
        lines = [
            (
                f"The {self.abbrev} is rated using the following "
                f"{len(self.anchors)}-point scale:"
            )
        ]
        for anchor in self.anchors:
            lines.append(f"  {anchor.value}. {anchor.label}")
        if is_package_installed("rich") and rich_print:
            text = Text()
            for i, line in enumerate(lines):
                if i == 0:
                    text.append(line + "\n", style="bold cyan")
                else:
                    text.append(line + "\n")
            return text

        return "\n".join(lines)

    def info_norms(self, *, rich_print: bool = True) -> str | Text:
        """Return information about available normative samples."""
        lines = [
            (
                f"The {self.abbrev} currently has {len(self.norms)} "
                "normative data set(s):\n"
            )
        ]
        for norm in self.norms:
            lines.append(f"{norm.sample_id}. {norm.size} {norm.population}")
            lines.append(f"   {norm.reference}")
            lines.append(f"   {norm.url}")
        if is_package_installed("rich") and rich_print:
            text = Text()
            for i, line in enumerate(lines):
                if i == 0:
                    text.append(line + "\n", style="bold cyan")
                else:
                    text.append(line + "\n")
            return text

        return "\n".join(lines)

    def score(
        self,
        data: pd.DataFrame,
        items: Iterable[str | int],
        prefix: str = "",
        suffix: str = "",
        *,
        na_rm: bool = True,
        append: bool = True,
    ) -> pd.DataFrame:
        """Compute mean scale scores for the instrument.

        Parameters
        ----------
        data
            DataFrame containing item-level data.
        items
            Iterable of item names or integer indices in `data`.
        prefix, suffix
            String affixes to add to resulting scale column names.
        na_rm
            If True, ignore missing values when computing means.
        append
            If True, append scores to `data`; else return only scores.
        """
        # Extract item data from the provided dataframe
        if all(
            isinstance(item, (int, np.integer, float, np.floating)) for item in items
        ):
            item_data = data.iloc[:, items].copy()
        elif all(isinstance(item, str) for item in items):
            item_data = data.loc[:, items].copy()
        else:
            msg = "All items in 'items' must be either strings or integers."
            raise TypeError(msg)

        scores = pd.DataFrame(index=data.index, columns=self.scale_abbrevs, dtype=float)
        for scale in self.scales:
            scale_items = [self.get_item(i).item_id - 1 for i in scale.items]
            scale_data = item_data.iloc[:, scale_items]

            scale_score = scale_data.mean(axis=1, skipna=na_rm)
            scores.loc[:, scale.abbrev] = scale_score

        scores.columns = [f"{prefix}{col}{suffix}" for col in scores.columns]

        if append:
            return pd.concat([data, scores], axis=1)
        return scores

    def norm_standardize(
        self,
        data: pd.DataFrame,
        sample_id: int,
        scales: Iterable[str | int] | None = None,
        prefix: str = "",
        suffix: str = "_z",
        *,
        append: bool = True,
    ) -> pd.DataFrame:
        """Standardize scale-level data using a normative sample.

        Parameters
        ----------
        data
            DataFrame containing at least circumplex scales.
        sample_id
            The ID of the normative sample to use for standardization.
        scales
            The variable names or column numbers for the scales in `data`. If None,
            all scales in the instrument are standardized.
        prefix
            Prefix to add to standardized scale names.
        suffix
            Suffix to add to standardized scale names.
        append
            If True, append standardized scales to `data`. If False,
            return only standardized scales.

        Returns
        -------
        DataFrame
            DataFrame with standardized scale-level data.

        Raises
        ------
        ValueError
            If `sample_id` is not found in the instrument's normative samples.
        """
        norm_sample = None
        for norm in self.norms:
            if norm.sample_id == sample_id:
                norm_sample = norm
                break
        if norm_sample is None:
            msg = (
                f"Normative sample with ID '{sample_id}' not found in instrument "
                f"'{self.abbrev}'."
            )
            raise ValueError(msg)
        if scales is None:
            scales = self.scale_abbrevs

        scores = pd.DataFrame(index=data.index, columns=[], dtype=float)
        for scale in scales:
            if isinstance(scale, (int, np.integer)):
                scale_abbrev = self.scales[scale].abbrev
            else:
                scale_abbrev = scale

            scale_data = data.loc[:, scale_abbrev]
            norm_stats = norm_sample.statistics
            mean_row = norm_stats.loc[norm_stats["scale"] == scale_abbrev]
            if mean_row.empty:
                msg = (
                    f"Normative statistics for scale '{scale_abbrev}' not found in "
                    f"sample ID '{sample_id}'."
                )
                raise ValueError(msg)
            mean = mean_row["mean"].to_numpy()[0]
            sd = mean_row["sd"].to_numpy()[0]

            standardized_score = (scale_data - mean) / sd
            scores[f"{prefix}{scale_abbrev}{suffix}"] = standardized_score

        if append:
            return pd.concat([data, scores], axis=1)
        return scores


# Global registry
_INSTRUMENTS: dict[str, Instrument] = {}


def register_instrument(abbrev: str, instrument: Instrument) -> None:
    """Register an instrument in the global registry."""
    _INSTRUMENTS[abbrev.lower()] = instrument


def get_instrument(abbrev: str) -> Instrument:
    """Retrieve an instrument by its abbreviation."""
    key = abbrev.lower()
    if key not in _INSTRUMENTS:
        available = ", ".join(_INSTRUMENTS.keys())
        msg = f"Instrument '{abbrev}' not found. Available instruments: {available}"
        raise ValueError(msg)
    return _INSTRUMENTS[key]


def show_instruments(*, rich_print: bool = True) -> None:
    """List all registered instrument abbreviations."""
    if is_package_installed("rich") and rich_print:
        table = Table(
            title=(
                "The circumplex package currently includes "
                f"{len(_INSTRUMENTS)} instruments"
            )
        )
        table.add_column("", no_wrap=True)
        table.add_column("Abbreviation", style="cyan", no_wrap=True)
        table.add_column("Name", style="magenta")
        for index, (abbrev, inst) in enumerate(_INSTRUMENTS.items(), start=1):
            table.add_row(str(index), abbrev.upper(), inst.name)
        if console is not None:  # safety if rich not actually available
            console.print(table)

    else:
        print(  # noqa: T201
            "The circumplex package currently includes "
            f"{len(_INSTRUMENTS)} instruments:"
        )
        for index, (abbrev, inst) in enumerate(_INSTRUMENTS.items(), start=1):
            print(f"  {index}. {abbrev.upper()}: {inst.name}")  # noqa: T201
