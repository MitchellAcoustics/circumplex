"""Shared pytest fixtures for circumplex tests."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from circumplex.data import load_dataset


@pytest.fixture(scope="session")
def jz2017_data() -> pd.DataFrame:
    """Load jz2017 dataset once per test session."""
    return load_dataset("jz2017")


@pytest.fixture(scope="session")
def aw2009_data() -> pd.DataFrame:
    """Load aw2009 dataset once per test session."""
    return load_dataset("aw2009")


def load_regression_fixture(fixture_name: str) -> dict[str, Any]:
    """Load a regression test fixture from JSON file.

    Parameters
    ----------
    fixture_name
        Name of fixture file (without .json extension)

    Returns
    -------
    Dictionary containing fixture data with keys:
        - dataset: Name of dataset to use
        - analysis_type: Type of analysis
        - seed: Random seed for reproducibility
        - input: Input parameters for analysis
        - expected: Expected results from R package

    """
    fixture_path = (
        Path(__file__).parent / "regression" / "fixtures" / f"{fixture_name}.json"
    )
    with fixture_path.open() as f:
        return json.load(f)


@pytest.fixture
def ssm_parameters_fixture() -> dict[str, Any]:
    """Load ssm_parameters regression test fixture."""
    return load_regression_fixture("ssm_parameters")


@pytest.fixture
def single_group_mean_fixture() -> dict[str, Any]:
    """Load single-group mean-based SSM regression test fixture."""
    return load_regression_fixture("ssm_single_group_mean")


@pytest.fixture
def multi_group_mean_fixture() -> dict[str, Any]:
    """Load multi-group mean-based SSM regression test fixture."""
    return load_regression_fixture("ssm_multi_group_mean")


@pytest.fixture
def multi_group_contrast_fixture() -> dict[str, Any]:
    """Load multi-group contrast SSM regression test fixture."""
    return load_regression_fixture("ssm_multi_group_contrast")


@pytest.fixture
def single_group_correlation_fixture() -> dict[str, Any]:
    """Load single-group correlation-based SSM regression test fixture."""
    return load_regression_fixture("ssm_single_group_correlation")


@pytest.fixture
def multi_group_correlation_fixture() -> dict[str, Any]:
    """Load multi-group correlation-based SSM regression test fixture."""
    return load_regression_fixture("ssm_multi_group_correlation")


@pytest.fixture
def measure_contrast_correlation_fixture() -> dict[str, Any]:
    """Load measure-contrast correlation-based SSM regression test fixture."""
    return load_regression_fixture("ssm_measure_contrast_correlation")


@pytest.fixture
def group_contrast_correlation_fixture() -> dict[str, Any]:
    """Load group-contrast correlation-based SSM regression test fixture."""
    return load_regression_fixture("ssm_group_contrast_correlation")


def assert_parameters_close(
    result: dict[str, float],
    expected: dict[str, float],
    *,
    rtol: float = 1e-3,
    convert_displacement: bool = True,
) -> None:
    """Assert SSM parameters match expected values within tolerance.

    Parameters
    ----------
    result
        Computed SSM parameters
    expected
        Expected SSM parameters from R (supports both fixture formats)
    rtol
        Relative tolerance for comparison (default: 0.1%)
    convert_displacement
        If True, convert displacement from radians to degrees for comparison

    """
    # Handle both fixture formats:
    # - ssm_parameters.json uses keys like "elevation", "x_value", etc.
    # - ssm_*_mean/correlation.json use keys like "e_est", "x_est", etc.
    elevation_key = "elevation" if "elevation" in expected else "e_est"
    x_key = "x_value" if "x_value" in expected else "x_est"
    y_key = "y_value" if "y_value" in expected else "y_est"
    amp_key = "amplitude" if "amplitude" in expected else "a_est"
    disp_key = "displacement" if "displacement" in expected else "d_est"
    fit_key = "fit" if "fit" in expected else "fit_est"

    # Standard parameters
    assert np.isclose(result["elevation"], expected[elevation_key], rtol=rtol), (
        "Elevation mismatch: "
        f"{result['elevation']:.3f} vs {expected[elevation_key]:.3f}"
    )
    assert np.isclose(result["x_value"], expected[x_key], rtol=rtol), (
        f"X-value mismatch: {result['x_value']:.3f} vs {expected[x_key]:.3f}"
    )
    assert np.isclose(result["y_value"], expected[y_key], rtol=rtol), (
        f"Y-value mismatch: {result['y_value']:.3f} vs {expected[y_key]:.3f}"
    )
    assert np.isclose(result["amplitude"], expected[amp_key], rtol=rtol), (
        f"Amplitude mismatch: {result['amplitude']:.3f} vs {expected[amp_key]:.3f}"
    )

    # Displacement needs special handling (convert from radians to degrees)
    if convert_displacement:
        displacement_deg = np.degrees(result["displacement"])
        assert np.isclose(displacement_deg, expected[disp_key], rtol=rtol), (
            "Displacement mismatch: "
            f"{displacement_deg:.3f}° vs {expected[disp_key]:.3f}°"
        )
    else:
        assert np.isclose(result["displacement"], expected[disp_key], rtol=rtol)

    assert np.isclose(result["fit"], expected[fit_key], rtol=rtol), (
        f"Fit mismatch: {result['fit']:.3f} vs {expected[fit_key]:.3f}"
    )


def assert_confidence_intervals_close(
    result: dict[str, tuple[float, float]],
    expected: dict[str, float],
    *,
    rtol: float = 1e-3,
    convert_displacement: bool = True,
) -> None:
    """Assert confidence intervals match expected values within tolerance.

    Parameters
    ----------
    result
        Computed confidence intervals (dict mapping parameter to (lower, upper))
    expected
        Expected CI values from R (dict with keys like 'e_lci', 'e_uci')
    rtol
        Relative tolerance for comparison (default: 0.1%)
    convert_displacement
        If True, convert displacement CIs from radians to degrees

    """
    # Map parameter names to their lower/upper CI keys in expected
    ci_params = {
        "elevation": ("e_lci", "e_uci"),
        "x_value": ("x_lci", "x_uci"),
        "y_value": ("y_lci", "y_uci"),
        "amplitude": ("a_lci", "a_uci"),
        "displacement": ("d_lci", "d_uci"),
    }

    for param, (lci_key, uci_key) in ci_params.items():
        if lci_key not in expected or uci_key not in expected:
            continue  # Skip if CIs not in expected results

        lower_computed, upper_computed = result[param]

        # Convert displacement from radians to degrees if needed
        if param == "displacement" and convert_displacement:
            lower_computed = np.degrees(lower_computed)
            upper_computed = np.degrees(upper_computed)

        lower_expected = expected[lci_key]
        upper_expected = expected[uci_key]

        assert np.isclose(lower_computed, lower_expected, rtol=rtol), (
            f"{param} lower CI mismatch: {lower_computed:.3f} vs {lower_expected:.3f}"
        )
        assert np.isclose(upper_computed, upper_expected, rtol=rtol), (
            f"{param} upper CI mismatch: {upper_computed:.3f} vs {upper_expected:.3f}"
        )
