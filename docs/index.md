# Welcome to Circumplex

[![PyPI version][pypi-badge]][pypi-link]
[![pre-commit][precommit-badge]][prek-link]
[![CI status][tests-badge]][tests-link]
[![Linting status][linting-badge]][linting-link]
[![Documentation status][documentation-badge]][documentation-link]
[![License][license-badge]](./LICENSE)

<!-- prettier-ignore-start -->
[pypi-badge]:               https://badge.fury.io/py/circumplex.svg
[pypi-link]:                https://pypi.org/project/circumplex/
[tests-badge]:              https://github.com/MitchellAcoustics/circumplex/actions/workflows/python-package.yml/badge.svg
[tests-link]:               https://github.com/MitchellAcoustics/circumplex/actions/workflows/python-package.yml
[precommit-badge]:          https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
[prek-link]:                https://prek.j178.dev/
[linting-badge]:            https://github.com/MitchellAcoustics/circumplex/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/MitchellAcoustics/circumplex/actions/workflows/linting.yml
[documentation-badge]:      https://github.com/MitchellAcoustics/circumplex/actions/workflows/docs.yml/badge.svg
[documentation-link]:       https://github.com/MitchellAcoustics/circumplex/actions/workflows/docs.yml
[license-badge]:            https://img.shields.io/badge/License-GPLv3-blue.svg

<!-- prettier-ignore-end -->

## Overview

![Image title](img/logo-light.png#only-light)
![Image title](img/logo-dark.png#only-dark)

_circumplex_ is a Python package for analyzing and visualizing circumplex data. It provides a set of tools for analyzing and visualizing circumplex data, following the Structural Summary Method. This project is a Python implementation based on the R [circumplex](https://github.com/jmgirard/circumplex) package. Our goal is to provide a similar functionality and experience for Python users.

This project is a Python implementation based on the R [circumplex](https://circumplex.jmgirard.com/) package. Our goal is to provide a similar functionality and experience for Python users.

!!! success "Massive update"
    The upcoming 0.3 release is the largest update to _circumplex_ so far. It
    introduces a modular package structure, expanded SSM and plotting workflows,
    built-in instrument and tidying utilities, stronger regression coverage against
    the R package, and refreshed documentation and CI/release infrastructure.
    See the [changelog](changelog.md) for release notes.

!!! note
    This project is still under development. We're working hard to make it as good as possible, but there may be bugs or missing features. If you find any issues, please let us know by submitting an issue on Github.

## Key Features

- **SSM Analysis**: Mean-based and correlation-based structural summary methods
- **Bootstrap Confidence Intervals**: Robust statistical inference for SSM parameters
- **Visualization**: Publication-ready circular plots and curve plots
- **Built-in Instruments**: IIP-SC, CSIG, IPIP-IPC with normative data
- **Flexible Instrument System**: Easy registration of custom circumplex measures
- **Data Tidying**: Ipsatization, scoring, and normative standardization

## Quick Start

### Installation

Install the latest release from PyPI:

```bash
pip install circumplex
```

### Basic Usage

```python
from circumplex import load_dataset, ssm_analyze, OCTANTS

# Load sample data
data = load_dataset('jz2017')

# Define scale columns and their angular positions
scales = ['PA', 'BC', 'DE', 'FG', 'HI', 'JK', 'LM', 'NO']
angles = OCTANTS  # [90, 135, 180, 225, 270, 315, 360, 45]

# Perform SSM analysis
results = ssm_analyze(data, scales=scales, angles=angles)

# View results
results.summary()

# Create visualizations
results.plot_circle()
results.plot_curve()
```

## Documentation

- **[API Reference](api/index.md)** - Complete API documentation
- **[Changelog](changelog.md)** - Release highlights and major updates
- **[GitHub Repository](https://github.com/MitchellAcoustics/circumplex)** - Source code and issue tracker

## Requirements

- Python 3.11, 3.12, or 3.13
- NumPy, Pandas, SciPy, Matplotlib, Seaborn

## Project Status

This project is in active development. Core SSM analysis functionality is implemented with full numerical parity to the R package (validated to 3+ decimal places). Additional features and documentation are being added continuously.

## Contributing

Contributions are welcome! Please see the [GitHub repository](https://github.com/MitchellAcoustics/circumplex) for guidelines.

## Acknowledgments

This project is developed in collaboration with the [Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University College London.

**Author:** Andrew Mitchell ([andrew.mitchell.research@gmail.com](mailto:andrew.mitchell.research@gmail.com))

**Based on:** The R circumplex package by Jeffrey Girard and colleagues

## License
