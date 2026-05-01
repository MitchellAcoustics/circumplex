# Changelog

All notable changes to _circumplex_ will be documented in this file.

## [Unreleased]

> [!IMPORTANT]
> This is the largest update to _circumplex_ so far and forms the basis of the upcoming 0.3 release. A major goal of this work has been to target feature parity and a familiar API for users of the fantastic R [circumplex](http://circumplex.jmgirard.com) package. The implementation is regression-tested against the R package to ensure equivalent results, and the Python package has been substantially reorganized, expanded, and hardened for public use.

### Added

- A modular package layout with dedicated `analysis`, `core`, `data`, `instruments`,
  `utils`, and `visualization` modules.
- Mean-based and correlation-based structural summary workflows with bootstrap-backed
  inference and richer plotting helpers.
- Built-in instrument definitions, instrument registration utilities, and tidying
  helpers for ipsatization, scoring, and normative standardization.
- Expanded API reference pages, tutorial content, examples, and regression fixtures
  to improve discoverability and validate parity with the reference R package.
- Added regression coverage against the R implementation to ensure the Python package
  produces equivalent results for supported structural summary workflows.

### Changed

- Reoriented the project around feature parity and a similar user experience to the
  R [circumplex](http://circumplex.jmgirard.com) package, so core workflows feel
  familiar to users moving between the two implementations.
- Refactored the public package around a clearer, more maintainable architecture for
  analysis, data loading, scoring, and plotting.
- Updated project infrastructure around `uv`, `tox`, `zensical`, and GitHub Actions to
  give the project a cleaner release, documentation, and CI story.
- Refreshed the README, documentation landing page, and documentation navigation to
  reflect the new package structure and upcoming release scope.

### Fixed

- Aligned package metadata, licensing, docs configuration, and release workflows so
  published artifacts and hosted documentation follow the same project configuration.
