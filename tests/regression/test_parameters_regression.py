"""Regression tests for SSM parameter calculations against R circumplex package.

These tests validate that the Python implementation produces numerically identical
results to the R package's ssm_parameters() function.
"""

import numpy as np
import pytest

from circumplex.core.parameters import ssm_parameters
from tests.conftest import assert_parameters_close


@pytest.mark.regression
class TestSSMParametersRegression:
    """Validate ssm_parameters() numerical parity with R package."""

    def test_ssm_parameters_basic(self, ssm_parameters_fixture):
        """Test ssm_parameters() matches R output exactly.

        This test validates the core parameter calculation function against
        the R package output using the aw2009 dataset (first row).

        Reference: R-circumplex/tests/testthat/test-ssm_analysis.R::test_that("ssm_parameters works")
        """  # noqa: E501
        # Extract input data from fixture
        scores = np.array(ssm_parameters_fixture["input"]["scores"])
        angles_deg = np.array(ssm_parameters_fixture["input"]["angles"])

        # Convert angles from degrees to radians (Python implementation uses radians)
        angles_rad = np.radians(angles_deg)

        # Calculate parameters
        result = ssm_parameters(scores, angles_rad)

        # Validate against expected R output
        expected = ssm_parameters_fixture["expected"]
        assert_parameters_close(result, expected, rtol=1e-3)

    def test_ssm_parameters_from_raw_data(self, aw2009_data):
        """Test ssm_parameters() on raw (non-standardized) dataset first row.

        This validates the function works with real pandas DataFrame data
        extracted the same way as in the R package test.
        Reference: R-circumplex/tests/testthat/test-ssm_analysis.R line 402-418
        """
        # Get first row of raw data (same as R test line 404)
        scales = ["PA", "BC", "DE", "FG", "HI", "JK", "LM", "NO"]
        scores = aw2009_data[scales].iloc[0].values

        # Use standard octant angles
        angles_deg = np.array([90, 135, 180, 225, 270, 315, 360, 45])
        angles_rad = np.radians(angles_deg)

        # Calculate parameters
        result = ssm_parameters(scores, angles_rad)

        # Expected values from R test (rounded to 2 decimals in R)
        # These are for RAW (non-standardized) scores
        # Reference: test-ssm_analysis.R lines 411-416
        assert np.isclose(result["elevation"], 0.43, atol=1e-2)
        assert np.isclose(result["x_value"], 1.25, atol=1e-2)
        assert np.isclose(result["y_value"], -1.31, atol=1e-2)
        assert np.isclose(result["amplitude"], 1.81, atol=1e-2)

        # Displacement in degrees
        displacement_deg = np.degrees(result["displacement"])
        assert np.isclose(displacement_deg, 313.71, atol=1e-1)

        assert np.isclose(result["fit"], 0.97, atol=1e-2)

    def test_ssm_parameters_second_row(self, aw2009_data):
        """Test ssm_parameters() on second row of aw2009 dataset.

        This is tested indirectly in the R package via ssm_score() function.
        Reference: test-ssm_analysis.R::test_that("ssm_score works")
        """
        # Get second row of data
        scales = ["PA", "BC", "DE", "FG", "HI", "JK", "LM", "NO"]
        scores = aw2009_data[scales].iloc[1].values

        # Use standard octant angles
        angles_deg = np.array([90, 135, 180, 225, 270, 315, 360, 45])
        angles_rad = np.radians(angles_deg)

        # Calculate parameters
        result = ssm_parameters(scores, angles_rad)

        # Expected values from R ssm_score test (rounded to 2 decimals)
        # Using absolute tolerance for comparison with rounded values
        assert np.isclose(result["elevation"], 0.23, atol=5e-3)
        assert np.isclose(result["x_value"], 1.42, atol=5e-3)
        assert np.isclose(result["y_value"], 0.51, atol=5e-3)
        assert np.isclose(result["amplitude"], 1.51, atol=5e-3)

        displacement_deg = np.degrees(result["displacement"])
        assert np.isclose(displacement_deg, 19.67, atol=5e-2)

        assert np.isclose(result["fit"], 0.92, atol=5e-3)

    def test_ssm_parameters_validates_input_length(self):
        """Test that ssm_parameters() requires matching array lengths."""
        scores = np.array([1.0, 2.0, 3.0])
        angles = np.array([0.0, np.pi / 2])  # Mismatched length

        # This should work fine - the function doesn't explicitly validate
        # But the calculation will fail with numpy broadcasting error
        with pytest.raises((ValueError, IndexError)):
            ssm_parameters(scores, angles)

    def test_ssm_parameters_zero_variance(self):
        """Test ssm_parameters() with zero variance (constant scores).

        When all scores are identical, variance is zero and fit should be 1.0.
        This tests the edge case handling in the fit calculation.
        """
        scores = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        angles = np.radians([90, 135, 180, 225, 270, 315, 360, 45])

        result = ssm_parameters(scores, angles)

        # With constant scores:
        # - Elevation should be the constant value
        # - x_value and y_value should be 0 (no variation around circle)
        # - Amplitude should be 0
        # - Fit should be 1.0 (perfect fit, handled by division by zero check)
        assert np.isclose(result["elevation"], 1.0)
        assert np.isclose(result["x_value"], 0.0, atol=1e-10)
        assert np.isclose(result["y_value"], 0.0, atol=1e-10)
        assert np.isclose(result["amplitude"], 0.0, atol=1e-10)
        assert result["fit"] == 1.0

    def test_ssm_parameters_displacement_range(self):
        """Test that displacement is always in [0, 2π) range."""
        # Test various angle configurations to ensure displacement wraps correctly
        test_cases = [
            # (scores, expected_displacement_deg)
            ([1, 0, 0, 0, 0, 0, 0, 0], 90),  # Peak at 90°
            ([0, 0, 0, 0, 1, 0, 0, 0], 270),  # Peak at 270°
            ([0, 0, 0, 0, 0, 0, 1, 0], 0),  # Peak at 360° (should wrap to ~0°)
        ]

        angles = np.radians([90, 135, 180, 225, 270, 315, 360, 45])

        for scores, expected_deg in test_cases:
            result = ssm_parameters(np.array(scores, dtype=float), angles)
            displacement_deg = np.degrees(result["displacement"])

            # Displacement should be in [0, 360) range
            # (allowing for floating point at boundary)
            assert 0 <= displacement_deg <= 360, (
                f"Displacement {displacement_deg} outside valid range for scores "
                f"{scores}"
            )

            # Note: 360° and 0° are equivalent due to circular nature
            # For the 360° angle case, we expect close to 0° or 360°
            if expected_deg == 0:
                # Allow either near 0 or near 360 due to modulo operation
                assert np.isclose(displacement_deg, 0, atol=1.0) or np.isclose(
                    displacement_deg, 360, atol=1.0
                ), f"Expected displacement near 0° or 360°, got {displacement_deg}°"
