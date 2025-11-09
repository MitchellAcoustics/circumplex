"""Circumplex: Analysis and visualization of circular data in Python.

This package provides tools for analyzing circular data using the Structural
Summary Method (SSM), particularly for interpersonal functioning, mood/affect,
and vocational preferences.

Main functions:

- ssm_analyze: Perform SSM analysis with bootstrap confidence intervals
- plot_circle: Plot SSM profiles on a circumplex circle
- plot_curve: Plot SSM fitted curves with observed scores
- plot_contrast: Plot SSM parameter contrasts between groups
- load_dataset: Load built-in example datasets
- OCTANTS, QUADRANTS, POLES: Standard angle sets

Examples
--------
>>> from circumplex import ssm_analyze, plot_circle
>>> from circumplex.data import load_dataset
>>>
>>> jz2017 = load_dataset('jz2017')
>>> results = ssm_analyze(jz2017, scales=['PA', 'BC', 'DE', 'FG',
...                                         'HI', 'JK', 'LM', 'NO'])
>>> print(results.results)
>>> fig = results.plot_circle()
>>> fig.savefig('profile.png')

"""

# Core analysis functions
# Utility functions
from circumplex import utils
from circumplex.analysis import ssm_analyze

# Data loading
from circumplex.data import load_dataset
from circumplex.utils.angles import OCTANTS, POLES, QUADRANTS, Degree, Radian, octants

# Visualization functions
from circumplex.visualization import plot_circle, plot_contrast, plot_curve

from ._version import __version__  # noqa: F401
from .instruments import (
    csig,
    get_instrument,
    iipsc,
    ipipipc,
    register_instrument,
    show_instruments,
)
from .utils.tidying_functions import ipsatize, norm_standardize, score

__all__ = [
    "OCTANTS",
    "POLES",
    "QUADRANTS",
    "Degree",
    "Radian",
    "csig",
    "get_instrument",
    "iipsc",
    "ipipipc",
    "ipsatize",
    "load_dataset",
    "norm_standardize",
    "octants",
    "plot_circle",
    "plot_contrast",
    "plot_curve",
    "register_instrument",
    "score",
    "show_instruments",
    "ssm_analyze",
    "utils",
]
