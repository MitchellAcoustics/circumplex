"""Regression tests for SSM analysis against R circumplex package.

These tests validate that the Python implementation of ssm_analyze() produces
numerically identical results to the R package, including all data preprocessing,
parameter calculations, and bootstrap confidence intervals.

Key understanding:
- Fixtures are generated from RAW data (e.g., aw2009 dataset as-is)
- ssm_analyze() must handle all preprocessing internally:
  1. Standardize/normalize circumplex scale scores (produces profile scores)
  2. Calculate SSM parameters from standardized scores
  3. Compute bootstrap confidence intervals
- The "scores" in fixtures are the STANDARDIZED profile scores, not raw data

NOTE: Most of these tests are now covered by tests/test_ssm_analyze.py
This file is kept for the unique tests (standardization, listwise/pairwise).
"""

import numpy as np
import pytest

from circumplex import ssm_analyze
from circumplex.utils.angles import OCTANTS


@pytest.mark.regression
class TestSSMAnalysisMeanBased:
    """Regression tests for mean-based SSM analysis."""

    def test_single_group_mean_based(self, aw2009_data, single_group_mean_fixture):
        """Test single-group mean-based SSM matches R output exactly.

        This is the most basic SSM analysis:
        - Input: Raw aw2009 dataset (5 rows x 8 scales)
        - Process: ssm_analyze() standardizes data and computes mean profile
        - Output: SSM parameters + bootstrap CIs for the group mean profile

        Reference:
        - R code: ssm_analyze(aw2009, scales = 1:8, boots = 2000)
        - R test: test-ssm_analysis.R::
            test_that("Single-group mean-based SSM results are correct")
        - Export script: export_test_fixtures.R lines 31-73

        Note: This test is now covered by
        test_ssm_analyze.py::test_ssm_single_group_mean
        """
        # Extract test parameters
        input_params = single_group_mean_fixture["input"]
        expected = single_group_mean_fixture["expected"]

        # Run analysis (angles should be in degrees)
        angles = OCTANTS
        result = ssm_analyze(
            aw2009_data,
            scales=input_params["scales"],
            angles=angles,
            boots=input_params["boots"],
            interval=input_params["interval"],
            listwise=input_params["listwise"],
            seed=single_group_mean_fixture["seed"],
        )

        # Get actual results from first row
        actual = result.results.iloc[0]

        # Validate parameter estimates (displacement already in degrees in SSM object)
        # Use both rtol and atol to handle fixture rounding to 3 decimals
        assert np.isclose(actual["e_est"], expected["e_est"], rtol=1e-3, atol=1e-3)
        assert np.isclose(actual["x_est"], expected["x_est"], rtol=1e-3, atol=1e-3)
        assert np.isclose(actual["y_est"], expected["y_est"], rtol=1e-3, atol=1e-3)
        assert np.isclose(actual["a_est"], expected["a_est"], rtol=1e-3, atol=1e-3)
        assert np.isclose(actual["d_est"], expected["d_est"], rtol=1e-3, atol=1e-3)
        assert np.isclose(actual["fit_est"], expected["fit_est"], rtol=1e-3, atol=1e-3)

        # Validate conf intervals (use relaxed tolerance for bootstrap variability)
        # With n=5, bootstrap CIs are highly variable
        assert np.isclose(actual["e_lci"], expected["e_lci"], rtol=0.20, atol=0.20)
        assert np.isclose(actual["e_uci"], expected["e_uci"], rtol=0.20, atol=0.20)
        assert np.isclose(actual["x_lci"], expected["x_lci"], rtol=0.20, atol=0.20)
        assert np.isclose(actual["x_uci"], expected["x_uci"], rtol=0.20, atol=0.20)
        assert np.isclose(actual["y_lci"], expected["y_lci"], rtol=0.20, atol=0.20)
        assert np.isclose(actual["y_uci"], expected["y_uci"], rtol=0.20, atol=0.20)
        assert np.isclose(actual["a_lci"], expected["a_lci"], rtol=0.20, atol=0.20)
        assert np.isclose(actual["a_uci"], expected["a_uci"], rtol=0.20, atol=0.20)
        assert np.isclose(actual["d_lci"], expected["d_lci"], rtol=0.20, atol=0.20)
        assert np.isclose(actual["d_uci"], expected["d_uci"], rtol=0.20, atol=0.20)

        # Validate standardized profile scores
        actual_scores = result.scores.iloc[0][input_params["scales"]].to_dict()
        for scale, expected_score in expected["scores"].items():
            # Use both rtol and atol for fixture rounding to 3 decimals
            assert np.isclose(
                actual_scores[scale], expected_score, rtol=1e-3, atol=1e-3
            ), f"Profile score mismatch for {scale}"

    def test_multi_group_mean_based(self, jz2017_data, multi_group_mean_fixture):
        """Test multi-group mean-based SSM matches R output.

        Tests grouping functionality:
        - Input: jz2017 dataset with Gender grouping variable
        - Process: Compute separate SSM profiles for each gender
        - Output: Two rows of results (Female, Male)

        Reference:
        - R code: ssm_analyze(jz2017, scales = 2:9, grouping = "Gender", boots = 2000)
        - R test: test-ssm_analysis.R::
            test_that("Multiple-group mean-based SSM results are correct")

        Note: This test is now covered by test_ssm_analyze.py::test_ssm_multi_group_mean
        """
        # Extract test parameters
        input_params = multi_group_mean_fixture["input"]
        expected = multi_group_mean_fixture["expected"]

        # Run analysis
        angles = OCTANTS
        result = ssm_analyze(
            jz2017_data,
            scales=input_params["scales"],
            angles=angles,
            grouping=input_params["grouping"],
            boots=input_params["boots"],
            interval=input_params["interval"],
            seed=multi_group_mean_fixture["seed"],
        )

        # Validate both groups
        assert len(result.results) == 2

        for idx in range(2):
            actual = result.results.iloc[idx]

            # Validate parameter estimates (use both rtol and atol for fixture rounding)
            assert np.isclose(
                actual["e_est"], expected["e_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["x_est"], expected["x_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["y_est"], expected["y_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["a_est"], expected["a_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["d_est"], expected["d_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["fit_est"], expected["fit_est"][idx], rtol=1e-3, atol=1e-3
            )

            # Validate confidence intervals (relaxed tolerance for bootstrap)
            assert np.isclose(
                actual["e_lci"], expected["e_lci"][idx], rtol=0.05, atol=0.05
            )
            assert np.isclose(
                actual["e_uci"], expected["e_uci"][idx], rtol=0.05, atol=0.05
            )
            assert np.isclose(
                actual["a_lci"], expected["a_lci"][idx], rtol=0.05, atol=0.05
            )
            assert np.isclose(
                actual["a_uci"], expected["a_uci"][idx], rtol=0.05, atol=0.05
            )

    def test_multi_group_contrast(self, jz2017_data, multi_group_contrast_fixture):
        """Test multi-group contrast SSM matches R output.

        Tests contrast functionality:
        - Input: jz2017 with Gender grouping + contrast=TRUE
        - Process: Compute profiles for each gender + Male - Female difference
        - Output: Three rows (Female, Male, Male - Female)

        Reference:
        - R code: ssm_analyze(jz2017, scales = 2:9, grouping = "Gender",
            contrast = TRUE)
        - R test: test-ssm_analysis.R::
            test_that("Multiple-group mean-based SSM contrast is correct")

        Note: This test is now covered by
        test_ssm_analyze.py::test_ssm_multi_group_mean_contrast
        """
        # Extract test parameters
        input_params = multi_group_contrast_fixture["input"]
        expected = multi_group_contrast_fixture["expected"]

        # Run analysis
        angles = OCTANTS
        result = ssm_analyze(
            jz2017_data,
            scales=input_params["scales"],
            angles=angles,
            grouping=input_params["grouping"],
            contrast=input_params["contrast"],
            boots=input_params["boots"],
            seed=multi_group_contrast_fixture["seed"],
        )

        # Validate all 3 rows (Female, Male, Male - Female)
        assert len(result.results) == 3

        for idx in range(3):
            actual = result.results.iloc[idx]

            # Validate parameter estimates (use both rtol and atol for fixture rounding)
            assert np.isclose(
                actual["e_est"], expected["e_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["x_est"], expected["x_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["y_est"], expected["y_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["a_est"], expected["a_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["d_est"], expected["d_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["fit_est"], expected["fit_est"][idx], rtol=1e-3, atol=1e-3
            )

            # Validate confidence intervals (relaxed tolerance for bootstrap)
            assert np.isclose(
                actual["e_lci"], expected["e_lci"][idx], rtol=0.05, atol=0.05
            )
            assert np.isclose(
                actual["e_uci"], expected["e_uci"][idx], rtol=0.05, atol=0.05
            )
            assert np.isclose(
                actual["a_lci"], expected["a_lci"][idx], rtol=0.05, atol=0.05
            )
            assert np.isclose(
                actual["a_uci"], expected["a_uci"][idx], rtol=0.05, atol=0.05
            )

        # Verify labels
        assert result.results.iloc[0]["Label"] == expected["labels"][0]
        assert result.results.iloc[1]["Label"] == expected["labels"][1]
        assert result.results.iloc[2]["Label"] == expected["labels"][2]


@pytest.mark.regression
class TestSSMAnalysisCorrelationBased:
    """Regression tests for correlation-based SSM analysis."""

    def test_single_group_correlation(
        self, jz2017_data, single_group_correlation_fixture
    ):
        """Test single-group correlation-based SSM matches R output.

        Tests correlation-based analysis:
        - Input: jz2017 with measures=["PARPD"] (external variable)
        - Process: Correlate each scale with PARPD, use correlations as profile
        - Output: SSM parameters describing correlation profile

        Key difference from mean-based:
        - Profile scores are correlations, not standardized means
        - Scores represent association strength with external measure

        Reference:
        - R code: ssm_analyze(jz2017, scales = 2:9, measures = "PARPD", boots = 2000)
        - R test: test-ssm_analysis.R::
            test_that("Single-group correlation-based SSM results are correct")

        Note: This test is now covered by
        test_ssm_analyze.py::test_ssm_single_group_correlation
        """
        # Extract test parameters
        input_params = single_group_correlation_fixture["input"]
        expected = single_group_correlation_fixture["expected"]

        # Run analysis
        angles = OCTANTS
        result = ssm_analyze(
            jz2017_data,
            scales=input_params["scales"],
            angles=angles,
            measures=input_params["measures"],
            boots=input_params["boots"],
            seed=single_group_correlation_fixture["seed"],
        )

        # Validate results
        actual = result.results.iloc[0]

        # Validate parameter estimates (use both rtol and atol for fixture rounding)
        assert np.isclose(actual["e_est"], expected["e_est"], rtol=1e-3, atol=1e-3)
        assert np.isclose(actual["x_est"], expected["x_est"], rtol=1e-3, atol=1e-3)
        assert np.isclose(actual["y_est"], expected["y_est"], rtol=1e-3, atol=1e-3)
        assert np.isclose(actual["a_est"], expected["a_est"], rtol=1e-3, atol=1e-3)
        # Note: displacement in fixture is rounded to 1 decimal for correlations
        assert np.isclose(actual["d_est"], expected["d_est"], rtol=0.1, atol=0.1)
        assert np.isclose(actual["fit_est"], expected["fit_est"], rtol=1e-3, atol=1e-3)

        # Validate confidence intervals (relaxed tolerance for bootstrap)
        assert np.isclose(actual["e_lci"], expected["e_lci"], rtol=0.05, atol=0.05)
        assert np.isclose(actual["e_uci"], expected["e_uci"], rtol=0.05, atol=0.05)
        assert np.isclose(actual["x_lci"], expected["x_lci"], rtol=0.05, atol=0.05)
        assert np.isclose(actual["x_uci"], expected["x_uci"], rtol=0.05, atol=0.05)
        assert np.isclose(actual["y_lci"], expected["y_lci"], rtol=0.05, atol=0.05)
        assert np.isclose(actual["y_uci"], expected["y_uci"], rtol=0.05, atol=0.05)

    def test_multi_group_correlation(
        self, jz2017_data, multi_group_correlation_fixture
    ):
        """Test multi-group correlation-based SSM matches R output.

        Combines correlation mode with grouping:
        - Separate correlation profiles for each gender

        Reference:
        - R test: test-ssm_analysis.R (multi-group correlation test)

        Note: This test is now covered by
        test_ssm_analyze.py::test_ssm_multi_group_correlation
        """
        # Extract test parameters
        input_params = multi_group_correlation_fixture["input"]
        expected = multi_group_correlation_fixture["expected"]

        # Run analysis
        angles = OCTANTS
        result = ssm_analyze(
            jz2017_data,
            scales=input_params["scales"],
            angles=angles,
            measures=input_params["measures"],
            grouping=input_params["grouping"],
            boots=input_params["boots"],
            seed=multi_group_correlation_fixture["seed"],
        )

        # Validate both groups
        assert len(result.results) == 2

        for idx in range(2):
            actual = result.results.iloc[idx]

            # Validate parameter estimates (use both rtol and atol for fixture rounding)
            assert np.isclose(
                actual["e_est"], expected["e_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["x_est"], expected["x_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["y_est"], expected["y_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["a_est"], expected["a_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["d_est"], expected["d_est"][idx], rtol=0.1, atol=0.1
            )
            assert np.isclose(
                actual["fit_est"], expected["fit_est"][idx], rtol=1e-3, atol=1e-3
            )

        # Verify labels
        assert result.results.iloc[0]["Label"] == expected["labels"][0]
        assert result.results.iloc[1]["Label"] == expected["labels"][1]

    def test_measure_contrast_correlation(
        self, jz2017_data, measure_contrast_correlation_fixture
    ):
        """Test measure-contrast correlation-based SSM matches R output.

        Tests contrasting two external measures:
        - Input: measures = ["ASPD", "NARPD"], contrast = TRUE
        - Output: Three profiles (ASPD, NARPD, NARPD - ASPD)

        Reference:
        - R code: ssm_analyze(jz2017, scales = 2:9, measures = c("ASPD", "NARPD"),
            contrast = TRUE)
        - R test: test-ssm_analysis.R::
            test_that("Measure-contrast correlation-based SSM results are correct")

        Note: This test is now covered by
        test_ssm_analyze.py::test_ssm_measure_contrast_correlation
        """
        # Extract test parameters
        input_params = measure_contrast_correlation_fixture["input"]
        expected = measure_contrast_correlation_fixture["expected"]

        # Run analysis
        angles = OCTANTS
        result = ssm_analyze(
            jz2017_data,
            scales=input_params["scales"],
            angles=angles,
            measures=input_params["measures"],
            contrast=input_params["contrast"],
            boots=input_params["boots"],
            seed=measure_contrast_correlation_fixture["seed"],
        )

        # Validate all 3 rows (ASPD, NARPD, NARPD - ASPD)
        assert len(result.results) == 3

        for idx in range(3):
            actual = result.results.iloc[idx]

            # Validate parameter estimates (use both rtol and atol for fixture rounding)
            assert np.isclose(
                actual["e_est"], expected["e_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["x_est"], expected["x_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["y_est"], expected["y_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["a_est"], expected["a_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["d_est"], expected["d_est"][idx], rtol=0.1, atol=0.1
            )
            assert np.isclose(
                actual["fit_est"], expected["fit_est"][idx], rtol=1e-3, atol=1e-3
            )

            # Validate confidence intervals (relaxed tolerance for bootstrap)
            assert np.isclose(
                actual["e_lci"], expected["e_lci"][idx], rtol=0.05, atol=0.05
            )
            assert np.isclose(
                actual["e_uci"], expected["e_uci"][idx], rtol=0.05, atol=0.05
            )
            assert np.isclose(
                actual["a_lci"], expected["a_lci"][idx], rtol=0.05, atol=0.05
            )
            assert np.isclose(
                actual["a_uci"], expected["a_uci"][idx], rtol=0.05, atol=0.05
            )

        # Verify labels
        assert result.results.iloc[0]["Label"] == expected["labels"][0]
        assert result.results.iloc[1]["Label"] == expected["labels"][1]
        assert result.results.iloc[2]["Label"] == expected["labels"][2]

        # Check scores for all three profiles
        score_names = input_params["scales"]
        actual_scores_aspd = result.scores.iloc[0][score_names].to_dict()
        actual_scores_narpd = result.scores.iloc[1][score_names].to_dict()
        actual_scores_contrast = result.scores.iloc[2][score_names].to_dict()

        # Use both rtol and atol for fixture rounding to 3 decimals
        for scale, expected_val in expected["scores_aspd"].items():
            assert np.isclose(
                actual_scores_aspd[scale], expected_val, rtol=1e-3, atol=1e-3
            )
        for scale, expected_val in expected["scores_narpd"].items():
            assert np.isclose(
                actual_scores_narpd[scale], expected_val, rtol=1e-3, atol=1e-3
            )
        for scale, expected_val in expected["scores_contrast"].items():
            assert np.isclose(
                actual_scores_contrast[scale], expected_val, rtol=1e-3, atol=1e-3
            )

    def test_group_contrast_correlation(
        self, jz2017_data, group_contrast_correlation_fixture
    ):
        """Test group-contrast correlation-based SSM matches R output.

        Tests contrasting groups in correlation mode:
        - Input: measures = "NARPD", grouping = "Gender", contrast = TRUE
        - Output: Three profiles (Female, Male, Male - Female)

        Reference:
        - R code: ssm_analyze(jz2017, scales = 2:9, measures = "NARPD",
          grouping = "Gender", contrast = TRUE)
        - R test: test-ssm_analysis.R::
          test_that("Group-contrast correlation-based SSM results are correct")

        Note: This test is now covered by
        test_ssm_analyze.py::test_ssm_group_contrast_correlation
        """
        # Extract test parameters
        input_params = group_contrast_correlation_fixture["input"]
        expected = group_contrast_correlation_fixture["expected"]

        # Run analysis
        angles = OCTANTS
        result = ssm_analyze(
            jz2017_data,
            scales=input_params["scales"],
            angles=angles,
            measures=input_params["measures"],
            grouping=input_params["grouping"],
            contrast=input_params["contrast"],
            boots=input_params["boots"],
            seed=group_contrast_correlation_fixture["seed"],
        )

        # Validate all 3 rows (Female, Male, Male - Female)
        assert len(result.results) == 3

        for idx in range(3):
            actual = result.results.iloc[idx]

            # Validate parameter estimates (use both rtol and atol for fixture rounding)
            assert np.isclose(
                actual["e_est"], expected["e_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["x_est"], expected["x_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["y_est"], expected["y_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["a_est"], expected["a_est"][idx], rtol=1e-3, atol=1e-3
            )
            assert np.isclose(
                actual["d_est"], expected["d_est"][idx], rtol=0.1, atol=0.1
            )
            assert np.isclose(
                actual["fit_est"], expected["fit_est"][idx], rtol=1e-3, atol=1e-3
            )

            # Validate confidence intervals (relaxed tolerance for bootstrap)
            assert np.isclose(
                actual["e_lci"], expected["e_lci"][idx], rtol=0.05, atol=0.05
            )
            assert np.isclose(
                actual["e_uci"], expected["e_uci"][idx], rtol=0.05, atol=0.05
            )
            assert np.isclose(
                actual["a_lci"], expected["a_lci"][idx], rtol=0.05, atol=0.05
            )
            assert np.isclose(
                actual["a_uci"], expected["a_uci"][idx], rtol=0.05, atol=0.05
            )

        # Verify labels
        assert result.results.iloc[0]["Label"] == expected["labels"][0]
        assert result.results.iloc[1]["Label"] == expected["labels"][1]
        assert result.results.iloc[2]["Label"] == expected["labels"][2]
