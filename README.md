# circumplex

<!-- markdownlint-disable MD033 -->
<img src="https://raw.githubusercontent.com/MitchellAcoustics/circumplex/main/docs/img/logo-dark.png" align="right" alt="" width="200" />

[![PyPI version][pypi-badge]][pypi-link]
[![pre-commit][precommit-badge]]
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
[linting-badge]:            https://github.com/MitchellAcoustics/circumplex/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/MitchellAcoustics/circumplex/actions/workflows/linting.yml
[documentation-badge]:      https://github.com/MitchellAcoustics/circumplex/actions/workflows/docs.yml/badge.svg
[documentation-link]:       https://github.com/MitchellAcoustics/circumplex/actions/workflows/docs.yml
[docs-site]:                https://drandrewmitchell.com/circumplex/
[license-badge]:            https://img.shields.io/badge/License-GPLv3-blue.svg

<!-- prettier-ignore-end -->

_circumplex_ is a Python package for analyzing and visualizing circumplex data. It provides a set of tools for analyzing and visualizing circumplex data, following the Structural Summary Method. This project is a Python implementation based on the R [circumplex](https://github.com/jmgirard/circumplex) package. Our goal is to provide a similar functionality and experience for Python users.

> [!IMPORTANT]
> **Massive update:** the upcoming 0.3 release is a substantial rewrite of
> _circumplex_. Highlights include a modular package architecture, expanded SSM and
> plotting workflows, built-in instrument and tidying utilities, stronger regression
> coverage against the R package, and a refreshed docs/CI/release toolchain.
> See the [changelog](CHANGELOG.md) for release notes.

<!-- markdownlint-disable MD028 -->
> [!WARNING]
> This project is still under development. We're working hard to make it as good as possible, but there may be bugs or missing features. If you find any issues, please let us know by submitting an issue on Github.

## Getting Started

To get started with _circumplex_, install it from PyPI:

```bash
pip install circumplex
```

## Documentation

This documentation is designed to help you understand and use _circumplex_ effectively. It's divided into several sections:

- **Docs Site**: The published documentation is available at [drandrewmitchell.com/circumplex][docs-site].
- **Tutorials**: Practical examples showing how to use our project in real-world scenarios.
- **API Reference**: Detailed information about our project's API.
- **Changelog**: Release highlights and migration context for the upcoming version.
- **Contribute**: Information on how you can contribute to our project.

## Contributing

We welcome contributions from the community. If you're interested in contributing, please get in touch or submit an issue on Github.

## License

This project is licensed under the GNU GPLv3 License. For more information, please see the `LICENSE` file.

## Project layout

```text
    .github/workflows/
        docs.yml
        linting.yml
        python-package.yml
    CHANGELOG.md
    docs/
        changelog.md
        index.md
        api/
        tutorials/
    examples/
    src/circumplex/
        analysis/
        core/
        instruments/
        visualization/
    pyproject.toml
    uv.lock
    zensical.toml
```
