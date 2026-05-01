"""IPIP Interpersonal Circumplex (IPIP-IPC) instrument definition.

Reference:
    Markey, P. M., & Markey, C. N. (2009). A brief assessment of the
    interpersonal circumplex: The IPIP-IPC. Assessment, 16(4), 352-361.
    https://doi.org/10.1177/1073191109340382
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
    InstrumentScale("PA", 90, (6, 14, 22, 30), "Assured-Dominant"),
    InstrumentScale("BC", 135, (7, 15, 23, 31), "Arrogant-Calculating"),
    InstrumentScale("DE", 180, (8, 16, 24, 32), "Cold-Hearted"),
    InstrumentScale("FG", 225, (1, 9, 17, 25), "Aloof-Introverted"),
    InstrumentScale("HI", 270, (2, 10, 18, 26), "Unassured-Submissive"),
    InstrumentScale("JK", 315, (3, 11, 19, 27), "Unassuming-Ingenuous"),
    InstrumentScale("LM", 360, (4, 12, 20, 28), "Warm-Agreeable"),
    InstrumentScale("NO", 45, (5, 13, 21, 29), "Gregarious-Extraverted"),
)

ANCHORS = (
    ResponseAnchor(1, "Very Inaccurate"),
    ResponseAnchor(2, "Moderately Inaccurate"),
    ResponseAnchor(3, "Neither Accurate nor Inaccurate"),
    ResponseAnchor(4, "Moderately Accurate"),
    ResponseAnchor(5, "Very Accurate"),
)

ITEMS = (
    ResponseItem(1, "Am quiet around strangers"),
    ResponseItem(2, "Speak softly"),
    ResponseItem(3, "Tolerate a lot from others"),
    ResponseItem(4, "Am interested in people"),
    ResponseItem(5, "Feel comfortable around people"),
    ResponseItem(6, "Demand to be the center of interest"),
    ResponseItem(7, "Cut others to pieces"),
    ResponseItem(8, "Believe people should fend for themselves"),
    ResponseItem(9, "Am a very private person"),
    ResponseItem(10, "Let others finish what they are saying"),
    ResponseItem(11, "Take things as they come"),
    ResponseItem(12, "Reassure others"),
    ResponseItem(13, "Start conversations"),
    ResponseItem(14, "Do most of the talking"),
    ResponseItem(15, "Contradict others"),
    ResponseItem(16, "Don't fall for sob-stories"),
    ResponseItem(17, "Don't talk a lot"),
    ResponseItem(18, "Seldom toot my own horn"),
    ResponseItem(19, "Think of others first"),
    ResponseItem(20, "Inquire about others' well-being"),
    ResponseItem(21, "Talk to a lot of different people at parties"),
    ResponseItem(22, "Speak loudly"),
    ResponseItem(23, "Snap at people"),
    ResponseItem(24, "Don't put a lot of thought into things"),
    ResponseItem(25, "Have little to say"),
    ResponseItem(26, "Dislike being the center of attention"),
    ResponseItem(27, "Seldom stretch the truth"),
    ResponseItem(28, "Get along well with others"),
    ResponseItem(29, "Love large parties"),
    ResponseItem(30, "Demand attention"),
    ResponseItem(31, "Have a sharp tongue"),
    ResponseItem(32, "Am not interested in other people's problems"),
)

norm_stats = pd.DataFrame(
    {
        "scale": ["PA", "BC", "DE", "FG", "HI", "JK", "LM", "NO"],
        "angle": [90, 135, 180, 225, 270, 315, 360, 45],
        "mean": [2.66, 2.27, 2.46, 2.68, 3.20, 3.64, 4.37, 3.64],
        "sd": [0.71, 0.69, 0.58, 0.79, 0.63, 0.58, 0.47, 0.78],
    }
)

NORMS = NormativeSample(
    sample_id=1,
    size=274,
    population="American college students",
    reference="Markey & Markey (2009)",
    url="https://doi.org/10.1177/1073191109340382",
    statistics=norm_stats,
)

INSTRUMENT = Instrument(
    name="IPIP Interpersonal Circumplex",
    abbrev="IPIP-IPC",
    construct="interpersonal traits",
    reference="Markey & Markey (2009)",
    url="https://doi.org/10.1177/1073191109340382",
    status="open-access",
    scales=SCALES,
    anchors=ANCHORS,
    items=ITEMS,
    norms=(NORMS,),
    prefix="",
    suffix="",
)


register_instrument("ipipipc", INSTRUMENT)
