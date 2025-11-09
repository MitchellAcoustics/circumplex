"""Circumplex Scales of Intergroup Goals (CSIG) instrument definition.

Reference:
    Lock, B. D. (2014). Circumplex scales of intergroup goals: An interpersonal circle
    model of goals for interactions between groups. Personality and Social
    Psychology Bulletin, 40(4), 433-449. https://kennethlocke.org/CSIG/CSIG.html
"""

import pandas as pd

from circumplex.instruments.models import (
    Instrument,
    InstrumentScale,
    NormativeSample,
    ResponseAnchor,
    ResponseItem,
    register_instrument,
)

# Define scales
SCALES = (
    InstrumentScale("PA", 90, items=(8, 16, 24, 32), label="Be authoritative"),
    InstrumentScale("BC", 135, items=(5, 12, 21, 29), label="Be tough"),
    InstrumentScale("DE", 180, items=(2, 10, 18, 26), label="Be self-protective"),
    InstrumentScale("FG", 225, items=(7, 15, 23, 31), label="Be wary"),
    InstrumentScale("HI", 270, items=(4, 12, 20, 28), label="Be conflict-avoidant"),
    InstrumentScale("JK", 315, items=(1, 9, 17, 25), label="Be cooperative"),
    InstrumentScale("LM", 360, items=(6, 14, 22, 30), label="Be understanding"),
    InstrumentScale("NO", 45, items=(3, 11, 19, 27), label="Be respected"),
)

ANCHORS = (
    ResponseAnchor(0, "It is not at all important that..."),
    ResponseAnchor(1, "It is somewhat important that..."),
    ResponseAnchor(2, "It is moderately important that..."),
    ResponseAnchor(3, "It is very important that..."),
    ResponseAnchor(4, "It is extremely important that..."),
)

ITEMS = (
    ResponseItem(1, "We are friendly"),
    ResponseItem(2, "We are the winners in any argument or dispute"),
    ResponseItem(3, "They respect what we have to say"),
    ResponseItem(4, "We avoid conflict"),
    ResponseItem(5, "We show that we can be tough"),
    ResponseItem(6, "We appreciate what they have to offer"),
    ResponseItem(7, "We let them fend for themselves"),
    ResponseItem(8, "We are assertive"),
    ResponseItem(9, "We celebrate their achievements"),
    ResponseItem(10, "We do whatever is in our best interest"),
    ResponseItem(11, "We get the chance to express our views"),
    ResponseItem(12, "They not get angry with us"),
    ResponseItem(13, "We not appear vulnerable"),
    ResponseItem(14, "We understand their point of view"),
    ResponseItem(15, "They stay out of our business"),
    ResponseItem(16, "We appear confident"),
    ResponseItem(17, "They feel we are all on the same team"),
    ResponseItem(18, "We are better than them"),
    ResponseItem(19, "They listen to what we have to say"),
    ResponseItem(20, "We not get into arguments"),
    ResponseItem(21, "We are aggressive if necessary"),
    ResponseItem(22, "We show concern for their welfare"),
    ResponseItem(23, "We not trust them"),
    ResponseItem(24, "We are decisive"),
    ResponseItem(25, "We are cooperative"),
    ResponseItem(26, "We keep our guard up"),
    ResponseItem(27, "They see us as responsible"),
    ResponseItem(28, "We not make them angry"),
    ResponseItem(29, "We not show our weaknesses"),
    ResponseItem(30, "We are able to compromise"),
    ResponseItem(31, "We not get entangled in their affairs"),
    ResponseItem(32, "They see us as capable"),
)


norm_stats = pd.DataFrame(
    {
        "scale": ["PA", "BC", "DE", "FG", "HI", "JK", "LM", "NO"],
        "angle": [90, 135, 180, 225, 270, 315, 360, 45],
        "mean": [2.96, 2.53, 2.02, 1.88, 2.24, 2.89, 2.97, 2.96],
        "sd": [0.68, 0.86, 0.88, 0.74, 0.90, 0.76, 0.71, 0.68],
    }
)

NORMS = NormativeSample(
    sample_id=1,
    size=665,
    population="MTurkers from US, Canada, and India about interactions between nations",
    reference="Lock (2014)",
    url="https://doi.org/10.1177/0146167213514280",
    statistics=norm_stats,
)

# Create Instrument
csig = Instrument(
    name="Circumplex Scales of Interpersonal Goals",
    abbrev="CSIG",
    construct="interpersonal intergroup goals",
    reference="Lock (2014)",
    url="https://doi.org/10.1177/0146167213514280",
    status="open-access",
    scales=SCALES,
    anchors=ANCHORS,
    items=ITEMS,
    norms=(NORMS,),
    prefix=(
        "In dealing with other groups, how important is it"
        "that we act or appear or are treated this way?"
    ),
    suffix="",
)

register_instrument("csig", csig)
