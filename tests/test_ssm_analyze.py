"""Regression tests for ssm_analyze against R package output.

These tests validate numerical parity with the R circumplex package
using pre-computed test fixtures from R analyses.

All tests updated to use the SSM object API instead of dict access.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from circumplex import ssm_analyze
from circumplex.data import load_dataset
from circumplex.utils.angles import OCTANTS

FIXTURES_DIR = Path(__file__).parent / "regression" / "fixtures"


def load_fixture(filename: str) -> dict:
    """Load a test fixture JSON file."""
    with (FIXTURES_DIR / filename).open() as f:
        return json.load(f)


def assert_close(
    actual: float, expected: float, rtol: float = 1e-3, atol: float = 1e-3
) -> None:
    """Assert two values are close to 3 decimal places."""
    if expected is None or np.isnan(expected):
        assert np.isnan(actual) or actual is None
    else:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def assert_close_ci(
    actual: float, expected: float, rtol: float = 0.05, atol: float = 0.05
) -> None:
    """Assert CI values are close (relaxed tolerance for bootstrap variability)."""
    # Bootstrap CIs can vary due to RNG implementation differences between R and Python.
    # Even with the same seed, R and Python generate different random sequences.
    # Default: 5% relative tolerance or 0.05 absolute tolerance for CIs.
    # Can be overridden for tests with very small samples or high variability.
    if expected is None or np.isnan(expected):
        assert np.isnan(actual) or actual is None
    else:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.regression
def test_ssm_single_group_mean() -> None:
    """Test single-group mean-based SSM analysis (aw2009 dataset).

    Note: This test uses a very small sample (n=5), which leads to high
    bootstrap variability. CI tolerances are wider than other tests.
    """
    fixture = load_fixture("ssm_single_group_mean.json")

    # Load data
    data = load_dataset("aw2009")

    # Run analysis (angles should be in degrees)
    angles = OCTANTS
    results = ssm_analyze(
        data,
        scales=fixture["input"]["scales"],
        angles=angles,
        boots=fixture["input"]["boots"],
        interval=fixture["input"]["interval"],
        listwise=fixture["input"]["listwise"],
        seed=fixture["seed"],
    )

    # Validate results
    expected = fixture["expected"]
    actual = results.results.iloc[0]

    # Check point estimates
    assert_close(actual["e_est"], expected["e_est"])
    assert_close(actual["x_est"], expected["x_est"])
    assert_close(actual["y_est"], expected["y_est"])
    assert_close(actual["a_est"], expected["a_est"])
    assert_close(actual["d_est"], expected["d_est"])
    assert_close(actual["fit_est"], expected["fit_est"])

    # Check confidence intervals (use extra-relaxed tolerance for small sample)
    # With n=5, bootstrap CIs are highly variable even with same algorithm
    assert_close_ci(actual["e_lci"], expected["e_lci"], rtol=0.20, atol=0.20)
    assert_close_ci(actual["e_uci"], expected["e_uci"], rtol=0.20, atol=0.20)
    assert_close_ci(actual["x_lci"], expected["x_lci"], rtol=0.20, atol=0.20)
    assert_close_ci(actual["x_uci"], expected["x_uci"], rtol=0.20, atol=0.20)
    assert_close_ci(actual["y_lci"], expected["y_lci"], rtol=0.20, atol=0.20)
    assert_close_ci(actual["y_uci"], expected["y_uci"], rtol=0.20, atol=0.20)
    assert_close_ci(actual["a_lci"], expected["a_lci"], rtol=0.20, atol=0.20)
    assert_close_ci(actual["a_uci"], expected["a_uci"], rtol=0.20, atol=0.20)

    # Check displacement CIs (may wrap around circle)
    assert_close_ci(actual["d_lci"], expected["d_lci"], rtol=0.20, atol=0.20)
    assert_close_ci(actual["d_uci"], expected["d_uci"], rtol=0.20, atol=0.20)

    # Check scores
    actual_scores = results.scores.iloc[0][fixture["input"]["scales"]].to_dict()
    for scale, expected_val in expected["scores"].items():
        assert_close(actual_scores[scale], expected_val)


@pytest.mark.regression
def test_ssm_multi_group_mean() -> None:
    """Test multi-group mean-based SSM analysis (jz2017 by Gender)."""
    fixture = load_fixture("ssm_multi_group_mean.json")

    # Load data
    data = load_dataset("jz2017")

    # Run analysis
    angles = OCTANTS
    results = ssm_analyze(
        data,
        scales=fixture["input"]["scales"],
        angles=angles,
        grouping=fixture["input"]["grouping"],
        boots=fixture["input"]["boots"],
        interval=fixture["input"]["interval"],
        seed=fixture["seed"],
    )

    # Validate results for both groups
    expected = fixture["expected"]
    actual_df = results.results

    # Check both groups
    for idx in range(2):
        actual = actual_df.iloc[idx]

        # Check point estimates
        assert_close(actual["e_est"], expected["e_est"][idx])
        assert_close(actual["x_est"], expected["x_est"][idx])
        assert_close(actual["y_est"], expected["y_est"][idx])
        assert_close(actual["a_est"], expected["a_est"][idx])
        assert_close(actual["d_est"], expected["d_est"][idx])
        assert_close(actual["fit_est"], expected["fit_est"][idx])

        # Check CIs (use relaxed tolerance for bootstrap variability)
        assert_close_ci(actual["e_lci"], expected["e_lci"][idx])
        assert_close_ci(actual["e_uci"], expected["e_uci"][idx])
        assert_close_ci(actual["a_lci"], expected["a_lci"][idx])
        assert_close_ci(actual["a_uci"], expected["a_uci"][idx])


@pytest.mark.regression
def test_ssm_multi_group_mean_contrast() -> None:
    """Test multi-group mean-based SSM with contrast."""
    fixture = load_fixture("ssm_multi_group_contrast.json")

    # Load data
    data = load_dataset("jz2017")

    # Run analysis
    angles = OCTANTS
    results = ssm_analyze(
        data,
        scales=fixture["input"]["scales"],
        angles=angles,
        grouping=fixture["input"]["grouping"],
        contrast=fixture["input"]["contrast"],
        boots=fixture["input"]["boots"],
        seed=fixture["seed"],
    )

    # Validate all 3 rows (Female, Male, Male - Female)
    expected = fixture["expected"]
    actual_df = results.results

    assert len(actual_df) == 3

    for idx in range(3):
        actual = actual_df.iloc[idx]

        # Check point estimates
        assert_close(actual["e_est"], expected["e_est"][idx])
        assert_close(actual["x_est"], expected["x_est"][idx])
        assert_close(actual["y_est"], expected["y_est"][idx])
        assert_close(actual["a_est"], expected["a_est"][idx])
        assert_close(actual["d_est"], expected["d_est"][idx])
        assert_close(actual["fit_est"], expected["fit_est"][idx])

        # Check some CIs (use relaxed tolerance for bootstrap variability)
        assert_close_ci(actual["e_lci"], expected["e_lci"][idx])
        assert_close_ci(actual["e_uci"], expected["e_uci"][idx])
        assert_close_ci(actual["a_lci"], expected["a_lci"][idx])
        assert_close_ci(actual["a_uci"], expected["a_uci"][idx])

    # Verify labels
    assert actual_df.iloc[0]["Label"] == expected["labels"][0]
    assert actual_df.iloc[1]["Label"] == expected["labels"][1]
    assert actual_df.iloc[2]["Label"] == expected["labels"][2]


@pytest.mark.regression
def test_ssm_single_group_correlation() -> None:
    """Test single-group correlation-based SSM analysis."""
    fixture = load_fixture("ssm_single_group_correlation.json")

    # Load data
    data = load_dataset("jz2017")

    # Run analysis
    angles = OCTANTS
    results = ssm_analyze(
        data,
        scales=fixture["input"]["scales"],
        angles=angles,
        measures=fixture["input"]["measures"],
        boots=fixture["input"]["boots"],
        seed=fixture["seed"],
    )

    # Validate results
    expected = fixture["expected"]
    actual = results.results.iloc[0]

    # Check point estimates
    assert_close(actual["e_est"], expected["e_est"])
    assert_close(actual["x_est"], expected["x_est"])
    assert_close(actual["y_est"], expected["y_est"])
    assert_close(actual["a_est"], expected["a_est"])
    # Note: displacement in fixture is rounded to 1 decimal for correlations
    assert_close(actual["d_est"], expected["d_est"], rtol=0.1)
    assert_close(actual["fit_est"], expected["fit_est"])

    # Check confidence intervals (use relaxed tolerance for bootstrap variability)
    assert_close_ci(actual["e_lci"], expected["e_lci"])
    assert_close_ci(actual["e_uci"], expected["e_uci"])
    assert_close_ci(actual["x_lci"], expected["x_lci"])
    assert_close_ci(actual["x_uci"], expected["x_uci"])
    assert_close_ci(actual["y_lci"], expected["y_lci"])
    assert_close_ci(actual["y_uci"], expected["y_uci"])


@pytest.mark.regression
def test_ssm_multi_group_correlation() -> None:
    """Test multi-group correlation-based SSM analysis."""
    fixture = load_fixture("ssm_multi_group_correlation.json")

    # Load data
    data = load_dataset("jz2017")

    # Run analysis
    angles = OCTANTS
    results = ssm_analyze(
        data,
        scales=fixture["input"]["scales"],
        angles=angles,
        measures=fixture["input"]["measures"],
        grouping=fixture["input"]["grouping"],
        boots=fixture["input"]["boots"],
        seed=fixture["seed"],
    )

    # Validate both groups
    expected = fixture["expected"]
    actual_df = results.results

    assert len(actual_df) == 2

    for idx in range(2):
        actual = actual_df.iloc[idx]

        # Check point estimates
        assert_close(actual["e_est"], expected["e_est"][idx])
        assert_close(actual["x_est"], expected["x_est"][idx])
        assert_close(actual["y_est"], expected["y_est"][idx])
        assert_close(actual["a_est"], expected["a_est"][idx])
        assert_close(actual["d_est"], expected["d_est"][idx], rtol=0.1)
        assert_close(actual["fit_est"], expected["fit_est"][idx])

    # Verify labels
    assert actual_df.iloc[0]["Label"] == expected["labels"][0]
    assert actual_df.iloc[1]["Label"] == expected["labels"][1]


@pytest.mark.regression
def test_ssm_measure_contrast_correlation() -> None:
    """Test measure-contrast correlation-based SSM analysis."""
    fixture = load_fixture("ssm_measure_contrast_correlation.json")

    # Load data
    data = load_dataset("jz2017")

    # Run analysis
    angles = OCTANTS
    results = ssm_analyze(
        data,
        scales=fixture["input"]["scales"],
        angles=angles,
        measures=fixture["input"]["measures"],
        contrast=fixture["input"]["contrast"],
        boots=fixture["input"]["boots"],
        seed=fixture["seed"],
    )

    # Validate all 3 rows (ASPD, NARPD, NARPD - ASPD)
    expected = fixture["expected"]
    actual_df = results.results

    assert len(actual_df) == 3

    for idx in range(3):
        actual = actual_df.iloc[idx]

        # Check point estimates
        assert_close(actual["e_est"], expected["e_est"][idx])
        assert_close(actual["x_est"], expected["x_est"][idx])
        assert_close(actual["y_est"], expected["y_est"][idx])
        assert_close(actual["a_est"], expected["a_est"][idx])
        assert_close(actual["d_est"], expected["d_est"][idx], rtol=0.1)
        assert_close(actual["fit_est"], expected["fit_est"][idx])

        # Check some CIs (use relaxed tolerance for bootstrap variability)
        assert_close_ci(actual["e_lci"], expected["e_lci"][idx])
        assert_close_ci(actual["e_uci"], expected["e_uci"][idx])
        assert_close_ci(actual["a_lci"], expected["a_lci"][idx])
        assert_close_ci(actual["a_uci"], expected["a_uci"][idx])

    # Verify labels
    assert actual_df.iloc[0]["Label"] == expected["labels"][0]
    assert actual_df.iloc[1]["Label"] == expected["labels"][1]
    assert actual_df.iloc[2]["Label"] == expected["labels"][2]

    # Check scores for all three profiles
    score_names = fixture["input"]["scales"]
    actual_scores_aspd = results.scores.iloc[0][score_names].to_dict()
    actual_scores_narpd = results.scores.iloc[1][score_names].to_dict()
    actual_scores_contrast = results.scores.iloc[2][score_names].to_dict()

    for scale, expected_val in expected["scores_aspd"].items():
        assert_close(actual_scores_aspd[scale], expected_val)
    for scale, expected_val in expected["scores_narpd"].items():
        assert_close(actual_scores_narpd[scale], expected_val)
    for scale, expected_val in expected["scores_contrast"].items():
        assert_close(actual_scores_contrast[scale], expected_val)


@pytest.mark.regression
def test_ssm_group_contrast_correlation() -> None:
    """Test group-contrast correlation-based SSM analysis."""
    fixture = load_fixture("ssm_group_contrast_correlation.json")

    # Load data
    data = load_dataset("jz2017")

    # Run analysis
    angles = OCTANTS
    results = ssm_analyze(
        data,
        scales=fixture["input"]["scales"],
        angles=angles,
        measures=fixture["input"]["measures"],
        grouping=fixture["input"]["grouping"],
        contrast=fixture["input"]["contrast"],
        boots=fixture["input"]["boots"],
        seed=fixture["seed"],
    )

    # Validate all 3 rows (Female, Male, Male - Female)
    expected = fixture["expected"]
    actual_df = results.results

    assert len(actual_df) == 3

    for idx in range(3):
        actual = actual_df.iloc[idx]

        # Check point estimates
        assert_close(actual["e_est"], expected["e_est"][idx])
        assert_close(actual["x_est"], expected["x_est"][idx])
        assert_close(actual["y_est"], expected["y_est"][idx])
        assert_close(actual["a_est"], expected["a_est"][idx])
        assert_close(actual["d_est"], expected["d_est"][idx], rtol=0.1)
        assert_close(actual["fit_est"], expected["fit_est"][idx])

        # Check some CIs (use relaxed tolerance for bootstrap variability)
        assert_close_ci(actual["e_lci"], expected["e_lci"][idx])
        assert_close_ci(actual["e_uci"], expected["e_uci"][idx])
        assert_close_ci(actual["a_lci"], expected["a_lci"][idx])
        assert_close_ci(actual["a_uci"], expected["a_uci"][idx])

    # Verify labels
    assert actual_df.iloc[0]["Label"] == expected["labels"][0]
    assert actual_df.iloc[1]["Label"] == expected["labels"][1]
    assert actual_df.iloc[2]["Label"] == expected["labels"][2]


@pytest.mark.unit
def test_ssm_analyze_basic_usage() -> None:
    """Test basic usage of ssm_analyze without bootstrap (fast test)."""
    data = load_dataset("aw2009")

    # Run with minimal bootstraps for speed
    results = ssm_analyze(data, scales=list(range(8)), boots=10, seed=123)

    # Check SSM object structure
    assert hasattr(results, "results")
    assert hasattr(results, "scores")
    assert hasattr(results, "details")
    assert hasattr(results, "type")

    # Check results DataFrame structure
    df = results.results
    assert "Label" in df.columns
    assert "Group" in df.columns
    assert "Measure" in df.columns
    assert "e_est" in df.columns
    assert "a_est" in df.columns
    assert "d_est" in df.columns
    assert "fit_est" in df.columns

    # Check that we have one row for single-group analysis
    assert len(df) == 1


@pytest.mark.unit
def test_ssm_analyze_with_grouping() -> None:
    """Test ssm_analyze with grouping variable."""
    data = load_dataset("jz2017")

    results = ssm_analyze(
        data, scales=list(range(1, 9)), grouping="Gender", boots=10, seed=123
    )

    # Should have 2 rows (Female, Male)
    assert len(results.results) == 2
    assert results.type == "mean"


@pytest.mark.unit
def test_ssm_analyze_with_contrast() -> None:
    """Test ssm_analyze with contrast=True."""
    data = load_dataset("jz2017")

    results = ssm_analyze(
        data,
        scales=list(range(1, 9)),
        grouping="Gender",
        contrast=True,
        boots=10,
        seed=123,
    )

    # Should have 3 rows (Female, Male, Male - Female)
    assert len(results.results) == 3
    assert "Male - Female" in results.results["Label"].values


@pytest.mark.unit
def test_ssm_analyze_correlation() -> None:
    """Test correlation-based SSM analysis."""
    data = load_dataset("jz2017")

    results = ssm_analyze(
        data, scales=list(range(1, 9)), measures="PARPD", boots=10, seed=123
    )

    # Check that it's correlation-based
    assert results.type == "correlation"
    assert len(results.results) == 1


@pytest.mark.unit
def test_ssm_analyze_invalid_contrast() -> None:
    """Test that invalid contrast raises ValueError."""
    data = load_dataset("jz2017")

    # Should fail: contrast with 3+ groups
    # Create a dummy 3-group dataset
    data_modified = data.copy()
    data_modified["Group3"] = ["A", "B", "C"] * (len(data) // 3) + ["A"] * (
        len(data) % 3
    )
    with pytest.raises(ValueError, match="Contrast can only be TRUE"):
        ssm_analyze(
            data_modified,
            scales=list(range(1, 9)),
            grouping="Group3",
            contrast=True,
            boots=10,
        )


@pytest.mark.unit
def test_ssm_analyze_custom_angles() -> None:
    """Test ssm_analyze with custom angles."""
    data = load_dataset("jz2017")

    # Use custom quadrant angles (4 scales)
    custom_angles = [90, 180, 270, 360]
    results = ssm_analyze(
        data, scales=["PA", "DE", "HI", "LM"], angles=custom_angles, boots=10, seed=123
    )

    assert len(results.results) == 1
    assert results.details.angles == custom_angles


@pytest.mark.unit
def test_plot_circle_basic() -> None:
    """Test that plot_circle() works and returns matplotlib Figure."""
    data = load_dataset("aw2009")
    results = ssm_analyze(data, scales=list(range(8)), boots=10, seed=123)

    fig = results.plot_circle()

    assert fig is not None
    assert hasattr(fig, "savefig")  # Check it's a matplotlib Figure
    plt.close(fig)


@pytest.mark.unit
def test_plot_curve_basic() -> None:
    """Test that plot_curve() works and returns matplotlib Figure."""
    data = load_dataset("aw2009")
    results = ssm_analyze(data, scales=list(range(8)), boots=10, seed=123)

    fig = results.plot_curve()

    assert fig is not None
    assert hasattr(fig, "savefig")
    plt.close(fig)


@pytest.mark.unit
def test_plot_contrast_basic() -> None:
    """Test that plot_contrast() works with contrast results."""
    data = load_dataset("jz2017")
    results = ssm_analyze(
        data,
        scales=list(range(1, 9)),
        grouping="Gender",
        contrast=True,
        boots=10,
        seed=123,
    )

    fig = results.plot_contrast()

    assert fig is not None
    assert hasattr(fig, "savefig")
    plt.close(fig)


@pytest.mark.unit
def test_plot_contrast_requires_contrast() -> None:
    """Test that plot_contrast() raises error without contrast results."""
    data = load_dataset("jz2017")
    results = ssm_analyze(
        data, scales=list(range(1, 9)), grouping="Gender", boots=10, seed=123
    )

    with pytest.raises(ValueError, match="requires contrast results"):
        results.plot_contrast()


@pytest.mark.unit
def test_plot_multiple_profiles() -> None:
    """Test plotting with multiple profiles."""
    data = load_dataset("jz2017")
    results = ssm_analyze(
        data,
        scales=list(range(1, 9)),
        measures=["ASPD", "NARPD"],
        boots=10,
        seed=123,
    )

    # All three plot types should work
    fig1 = results.plot_circle()
    fig2 = results.plot_curve()

    assert fig1 is not None
    assert fig2 is not None

    plt.close(fig1)
    plt.close(fig2)
