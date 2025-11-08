import numpy as np
import pandas as pd
import pytest

from circumplex import SSMResults, ssm_analysis, utils

SCALES = ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8")
OCTANTS = utils.OCTANTS


def fixed_data(angles=OCTANTS, ampl=0.5, disp=180, elev: float = 0):
    return utils.cosine_form(np.deg2rad(angles), ampl, np.deg2rad(disp), elev)


def generate_circumplex_data(angles, amplitude, displacement, elevation):
    """Generate circumplex data with known parameters."""
    scores = elevation + amplitude * np.cos(
        np.deg2rad(angles) - np.deg2rad(displacement)
    )
    return scores


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "PA": np.random.rand(100),
            "BC": np.random.rand(100),
            "DE": np.random.rand(100),
            "FG": np.random.rand(100),
            "HI": np.random.rand(100),
            "JK": np.random.rand(100),
            "LM": np.random.rand(100),
            "NO": np.random.rand(100),
            "measure1": np.random.rand(100),
            "measure2": np.random.rand(100),
            "group": np.random.choice(["A", "B"], size=100),
        }
    )


@pytest.fixture
def scales():
    return ["PA", "BC", "DE", "FG", "HI", "JK", "LM", "NO"]


@pytest.fixture
def angles():
    return OCTANTS


def test_ssm_analyze_basic(sample_data, scales, angles):
    result = ssm_analysis.ssm_analyze(sample_data, scales, angles, boots=50)
    assert isinstance(result, SSMResults)
    assert len(result.results) == 1  # One row for overall results


def test_ssm_analyze_with_measures(sample_data, scales, angles):
    result = ssm_analysis.ssm_analyze(
        sample_data, scales, angles, measures=["measure1", "measure2"], boots=50
    )
    assert isinstance(result, SSMResults)
    assert len(result.results) == 2  # One row for each measure


def test_ssm_analyze_with_grouping(sample_data, scales, angles):
    result = ssm_analysis.ssm_analyze(
        sample_data, scales, angles, grouping="group", boots=50
    )
    assert isinstance(result, SSMResults)
    assert len(result.results) == 2  # One row for each group


def test_ssm_analyze_with_contrast(sample_data, scales, angles):
    result = ssm_analysis.ssm_analyze(
        sample_data, scales, angles, grouping="group", contrast="test", boots=50
    )
    assert isinstance(result, SSMResults)
    assert len(result.results) == 1  # One row for the contrast


def test_ssm_parameters_return(angles):
    scores = np.array([0.5, 0.7, 0.3, 0.2, 0.8, 0.6, 0.4, 0.9])
    angles_rad = np.deg2rad(angles)
    params = ssm_analysis.ssm_parameters(scores, angles_rad)
    assert len(params) == 6  # Should return 6 parameters


def test_group_parameters(angles):
    scores = np.array(
        [
            [0.5, 0.7, 0.3, 0.2, 0.8, 0.6, 0.4, 0.9],
            [0.4, 0.6, 0.2, 0.3, 0.7, 0.5, 0.3, 0.8],
        ]
    )
    angles_rad = np.deg2rad(angles)
    params = ssm_analysis.group_parameters(scores, angles_rad)
    assert len(params) == 12  # Should return 12 parameters (6 for each group)


def test_ssm_bootstrap(sample_data, scales, angles):
    bs_input = sample_data[scales + ["group"]]

    def bs_function(data, index, angles, contrast, listwise):
        return np.random.rand(6)  # Mock function

    result = ssm_analysis.ssm_bootstrap(
        bs_input,
        bs_function,
        np.deg2rad(angles),
        100,
        0.95,
        "none",
        True,
        bs_input["group"],
    )
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == 18  # 6 parameters * 3 (est, lci, uci)


def test_invalid_input(sample_data, angles):
    with pytest.raises((AssertionError, ValueError)):
        ssm_analysis.ssm_analyze(sample_data, ["invalid_scale"], angles, boots=50)


def test_missing_data(sample_data, scales, angles):
    data_with_nan = sample_data.copy()
    data_with_nan.loc[0, "PA"] = np.nan
    result = ssm_analysis.ssm_analyze(data_with_nan, scales, angles, boots=50)
    assert isinstance(result, SSMResults)


def test_ssm_parameters():
    # elev, xval, yval, ampl, disp, r2
    fix_data = fixed_data()
    fixed_ssm = ssm_analysis.ssm_parameters(fix_data, utils.OCTANTS)
    np.testing.assert_allclose(fixed_ssm, (0.0, -0.5, 0, 0.5, 180, 1.0), atol=1e-4)

    fix_data = fixed_data(ampl=0.5, disp=45, elev=0)
    fixed_ssm = ssm_analysis.ssm_parameters(fix_data, utils.OCTANTS)
    np.testing.assert_allclose(
        fixed_ssm, (0.0, 0.35355, 0.35355, 0.5, 45, 1.0), atol=1e-4
    )

    fix_data = fixed_data(ampl=0.3, disp=90, elev=0.1)
    fixed_ssm = ssm_analysis.ssm_parameters(fix_data, utils.OCTANTS)
    np.testing.assert_allclose(fixed_ssm, (0.1, 0.0, 0.3, 0.3, 90, 1.0), atol=1e-4)


@pytest.mark.parametrize(
    "amplitude, displacement, elevation",
    [
        (1.0, 3, 0.5),  # 3 degrees
        (1.0, 90, 0.5),  # 90 degrees
        (1.0, 180, 0.5),  # 180 degrees
        (1.0, 270, 0.5),  # 270 degrees
        (1.0, 45, 0.5),  # 45 degrees
        (2.0, 30, 1.0),  # Larger amplitude
        (0.2, 60, 0.1),  # Smaller amplitude
        (1.5, 135, 0.5),  # Arbitrary values
    ],
)
def test_ssm_parameters_correctness(amplitude, displacement, elevation):
    scores = generate_circumplex_data(OCTANTS, amplitude, displacement, elevation)

    # Calculate expected x and y values
    expected_x = amplitude * np.cos(np.deg2rad(displacement))
    expected_y = amplitude * np.sin(np.deg2rad(displacement))
    expected_r2 = 1.0  # Perfect fit for simulated data

    # Run ssm_parameters
    params = ssm_analysis.ssm_parameters(scores, OCTANTS)

    # Extract results
    (
        result_elevation,
        result_x,
        result_y,
        result_amplitude,
        result_displacement,
        result_r2,
    ) = params

    result_displacement = result_displacement % 360  # Normalize to 0-360 degrees

    # Assert correctness with some tolerance for floating-point precision
    np.testing.assert_allclose(result_elevation, elevation, atol=1e-7)
    np.testing.assert_allclose(result_x, expected_x, atol=1e-7)
    np.testing.assert_allclose(result_y, expected_y, atol=1e-7)
    np.testing.assert_allclose(result_amplitude, amplitude, atol=1e-7)
    np.testing.assert_allclose(result_displacement, displacement, atol=1e-7)
    np.testing.assert_allclose(result_r2, expected_r2, atol=1e-7)
