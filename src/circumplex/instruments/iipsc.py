"""Inventory of Interpersonal Problems Short Circumplex (IIP-SC) instrument definition.

Reference:
    Soldz, S., Budman, S., Demby, A., & Merry, J. (1995). A short form of the Inventory
    of Interpersonal Problems Circumplex Scales. Assessment, 2(1), 53-63.
    https://doi.org/10.1080/00223890802388665
"""

import numpy as np
import pandas as pd

from circumplex.instruments.models import (
    Instrument,
    InstrumentScale,
    NormativeSample,
    ResponseAnchor,
    ResponseItem,
    register_instrument,
)

SCALES = (
    InstrumentScale("PA", 90, items=(1, 9, 17, 25), label="Domineering"),
    InstrumentScale("BC", 135, items=(2, 10, 18, 26), label="Vindictive"),
    InstrumentScale("DE", 180, items=(3, 11, 19, 27), label="Cold"),
    InstrumentScale("FG", 225, items=(4, 12, 20, 28), label="Socially avoidant"),
    InstrumentScale("HI", 270, items=(5, 13, 21, 29), label="Nonassertive"),
    InstrumentScale("JK", 315, items=(6, 14, 22, 30), label="Exploitable"),
    InstrumentScale("LM", 360, items=(7, 15, 23, 31), label="Overly nurturant"),
    InstrumentScale("NO", 45, items=(8, 16, 24, 32), label="Intrusive"),
)

ANCHORS = (
    ResponseAnchor(0, "Not at all"),
    ResponseAnchor(1, "Somewhat"),
    ResponseAnchor(2, "Moderately"),
    ResponseAnchor(3, "Very"),
    ResponseAnchor(4, "Extremely"),
)

ITEMS = (
    ResponseItem(1, "...point of view..."),
    ResponseItem(2, "...supportive of another..."),
    ResponseItem(3, "...show affection to..."),
    ResponseItem(4, "...join in on..."),
    ResponseItem(5, "...stop bothering me..."),
    ResponseItem(6, "...I am angry..."),
    ResponseItem(7, "...my own welfare..."),
    ResponseItem(8, "...keep things private..."),
    ResponseItem(9, "...too aggressive toward..."),
    ResponseItem(10, "...another person's happiness..."),
    ResponseItem(11, "...feeling of love..."),
    ResponseItem(12, "...introduce myself to..."),
    ResponseItem(13, "...confront people with..."),
    ResponseItem(14, "...assertive without worrying..."),
    ResponseItem(15, "...please other people..."),
    ResponseItem(16, "...open up to..."),
    ResponseItem(17, "...control other people..."),
    ResponseItem(18, "...too suspicious of..."),
    ResponseItem(19, "...feel close to..."),
    ResponseItem(20, "...socialize with other..."),
    ResponseItem(21, "...assertive with another..."),
    ResponseItem(22, "...too easily persuaded..."),
    ResponseItem(23, "...other people's needs..."),
    ResponseItem(24, "...noticed too much..."),
    ResponseItem(25, "...argue with other..."),
    ResponseItem(26, "...revenge against people..."),
    ResponseItem(27, "...at a distance..."),
    ResponseItem(28, "...get together socially..."),
    ResponseItem(29, "...to be firm..."),
    ResponseItem(30, "...people take advantage..."),
    ResponseItem(31, "...another person's misery..."),
    ResponseItem(32, "...tell personal things..."),
)

norm_1_stats = pd.DataFrame(
    {
        "scale": ["PA", "BC", "DE", "FG", "HI", "JK", "LM", "NO"],
        "angles": [90, 135, 180, 225, 270, 315, 360, 45],
        "mean": np.array([3.04, 3.17, 3.60, 4.19, 5.68, 5.54, 5.86, 4.10]) / 4,
        "sd": np.array([2.64, 2.76, 3.42, 3.79, 3.66, 3.41, 3.30, 3.20]) / 4,
    }
)

NORM1 = NormativeSample(
    sample_id=1,
    size=872,
    population="American college students",
    reference="Hopwood, Pincus, DeMoor, & Koonce (2011)",
    url="https://doi.org/10.1080/00223890802388665",
    statistics=norm_1_stats,
)

norm_2_stats = pd.DataFrame(
    {
        "scale": ["PA", "BC", "DE", "FG", "HI", "JK", "LM", "NO"],
        "angles": [90, 135, 180, 225, 270, 315, 360, 45],
        "mean": [0.99, 0.97, 1.30, 1.33, 1.81, 1.92, 2.14, 1.43],
        "sd": [0.82, 0.85, 1.07, 0.98, 0.89, 0.89, 0.90, 1.05],
    }
)

NORM2 = NormativeSample(
    sample_id=2,
    size=106,
    population="American psychiatric outpatients",
    reference="Soldz, Budman, Demby, & Merry (1995)",
    url="https://doi.org/10.1177/1073191195002001006",
    statistics=norm_2_stats,
)

iipsc = Instrument(
    name="Inventory of Interpersonal Problems Short Circumplex",
    abbrev="IIP-SC",
    construct="interpersonal problems",
    reference="Soldz, Budman, Demby, & Merry (1995)",
    url="https://doi.org/10.1177/1073191195002001006",
    status="partial text",
    scales=SCALES,
    anchors=ANCHORS,
    items=ITEMS,
    norms=(NORM1, NORM2),
    prefix="",
    suffix="",
)

register_instrument("iipsc", iipsc)
