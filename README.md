# circumplex

<!-- markdownlint-disable MD033 -->
<img src="https://raw.githubusercontent.com/MitchellAcoustics/circumplex/main/docs/img/logo-dark.png" align="right" alt="" width="200" />

[![PyPI version](https://badge.fury.io/py/circumplex.svg)](https://badge.fury.io/py/circumplex)
[![Documentation Status](https://readthedocs.org/projects/circumplex/badge/?version=latest)](https://circumplex.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status][tests-badge]][tests-link]
[![Linting status][linting-badge]][linting-link]
[![Documentation status][documentation-badge]][documentation-link]
[![License][license-badge]](./LICENSE)

<!-- prettier-ignore-start -->
[tests-badge]:              https://github.com/MitchellAcoustics/circumplex/actions/workflows/tests.yml/badge.svg
[tests-link]:               https://github.com/MitchellAcoustics/circumplex/actions/workflows/tests.yml
[linting-badge]:            https://github.com/MitchellAcoustics/circumplex/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/MitchellAcoustics/circumplex/actions/workflows/linting.yml
[documentation-badge]:      https://github.com/MitchellAcoustics/circumplex/actions/workflows/docs.yml/badge.svg
[documentation-link]:       https://github.com/MitchellAcoustics/circumplex/actions/workflows/docs.yml
[license-badge]:            https://img.shields.io/badge/License-GPLv3-blue.svg

<!-- prettier-ignore-end -->

_circumplex_ is a Python package for analyzing and visualizing circumplex data. It provides a set of tools for analyzing and visualizing circumplex data, following the Structural Summary Method. This project is a Python implementation based on the R [circumplex](https://github.com/jmgirard/circumplex) package. Our goal is to provide a similar functionality and experience for Python users.

> [!WARNING]
> This project is still under development. We're working hard to make it as good as possible, but there may be bugs or missing features. If you find any issues, please let us know by submitting an issue on Github.

## Getting Started

To get started with _circumplex_, you'll need to install it first. You can do this by running the following command:

```bash
pip install circumplex
```

## Documentation

This documentation is designed to help you understand and use _circumplex_ effectively. It's divided into several sections:

- **Tutorials**: Practical examples showing how to use our project in real-world scenarios.
- **API Reference**: Detailed information about our project's API.
- **Contribute**: Information on how you can contribute to our project.

## Contributing

We welcome contributions from the community. If you're interested in contributing, please get in touch or submit an issue on Github.

## License

This project is licensed under the GNU GPLv3 License. For more information, please see the `LICENSE` file.

## Project layout

```text
    mkdocs.yml     # The configuration file.
    docs/
        index.md   # The documentation homepage.
        about.md   # The about page.
        LICENSE # The license page.
        tutorials/ # Tutorial pages.
            Introduction to SSM Analysis.ipynb
        ...        # Other markdown pages, images and other files.
    src/circumplex/
        circumplex.py
```
