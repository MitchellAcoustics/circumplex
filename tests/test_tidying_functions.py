"""Tests for data tidying and instrument scoring utility functions."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from circumplex.instruments import Instrument
from circumplex.utils.tidying_functions import ipsatize, norm_standardize, score


class TestIpsatize:
    """Tests for the ipsatize function."""

    def test_ipsatize_basic_with_string_items(self):
        """Test basic ipsatization with string column names."""
        data = pd.DataFrame(
            {
                "item1": [1, 2, 3],
                "item2": [2, 3, 4],
                "item3": [3, 4, 5],
            }
        )
        result = ipsatize(data, ["item1", "item2", "item3"], append=False)

        # Expected: each row centered to mean of 0
        expected = pd.DataFrame(
            {
                "item1_i": [-1.0, -1.0, -1.0],
                "item2_i": [0.0, 0.0, 0.0],
                "item3_i": [1.0, 1.0, 1.0],
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_ipsatize_with_integer_indices(self):
        """Test ipsatization with integer column indices."""
        data = pd.DataFrame(
            {
                "item1": [1, 2, 3],
                "item2": [2, 3, 4],
                "item3": [3, 4, 5],
            }
        )
        result = ipsatize(data, [0, 1, 2], append=False)

        expected = pd.DataFrame(
            {
                "0_i": [-1.0, -1.0, -1.0],
                "1_i": [0.0, 0.0, 0.0],
                "2_i": [1.0, 1.0, 1.0],
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_ipsatize_with_prefix_suffix(self):
        """Test ipsatization with custom prefix and suffix."""
        data = pd.DataFrame(
            {
                "item1": [1, 2],
                "item2": [2, 3],
            }
        )
        result = ipsatize(
            data, ["item1", "item2"], prefix="pre_", suffix="_post", append=False
        )

        assert "pre_item1_post" in result.columns
        assert "pre_item2_post" in result.columns

    def test_ipsatize_append_true(self):
        """Test that append=True keeps original columns."""
        data = pd.DataFrame(
            {
                "id": [1, 2],
                "item1": [1, 2],
                "item2": [2, 3],
            }
        )
        result = ipsatize(data, ["item1", "item2"], append=True)

        assert "id" in result.columns
        assert "item1" in result.columns
        assert "item2" in result.columns
        assert "item1_i" in result.columns
        assert "item2_i" in result.columns

    def test_ipsatize_append_false(self):
        """Test that append=False returns only ipsatized columns."""
        data = pd.DataFrame(
            {
                "id": [1, 2],
                "item1": [1, 2],
                "item2": [2, 3],
            }
        )
        result = ipsatize(data, ["item1", "item2"], append=False)

        assert "id" not in result.columns
        assert "item1" not in result.columns
        assert "item2" not in result.columns
        assert "item1_i" in result.columns
        assert "item2_i" in result.columns

    def test_ipsatize_with_na_values_na_rm_true(self):
        """Test ipsatization with NA values when na_rm=True."""
        data = pd.DataFrame(
            {
                "item1": [1, 2, np.nan],
                "item2": [2, 3, 4],
                "item3": [3, np.nan, 5],
            }
        )
        result = ipsatize(data, ["item1", "item2", "item3"], na_rm=True, append=False)

        # First row: mean of [1, 2, 3] = 2
        assert np.isclose(result.iloc[0, 0], -1.0)
        # Second row: mean of [2, 3] = 2.5 (ignoring NaN)
        assert np.isclose(result.iloc[1, 0], -0.5)

    def test_ipsatize_with_na_values_na_rm_false(self):
        """Test ipsatization with NA values when na_rm=False."""
        data = pd.DataFrame(
            {
                "item1": [1, 2, np.nan],
                "item2": [2, 3, 4],
            }
        )
        result = ipsatize(data, ["item1", "item2"], na_rm=False, append=False)

        # Third row should have NaN mean
        assert pd.isna(result.iloc[2, 0])

    def test_ipsatize_invalid_data_type(self):
        """Test that non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="Input 'data' must be a pandas DataFrame"):
            ipsatize([1, 2, 3], ["item1"])  # type: ignore[invalid-argument-type]

    def test_ipsatize_invalid_items_type(self):
        """Test that non-iterable items raises TypeError."""
        data = pd.DataFrame({"item1": [1, 2]})
        with pytest.raises(TypeError, match="Input 'items' must be a sequence"):
            ipsatize(data, "item1")

    def test_ipsatize_invalid_column_names(self):
        """Test that invalid column names raise ValueError."""
        data = pd.DataFrame({"item1": [1, 2]})
        with pytest.raises(
            ValueError, match="All items in 'items' must be valid column names"
        ):
            ipsatize(data, ["item1", "invalid"])

    def test_ipsatize_invalid_column_indices(self):
        """Test that invalid column indices raise ValueError."""
        data = pd.DataFrame({"item1": [1, 2]})
        with pytest.raises(
            ValueError, match="All items in 'items' must be valid indices"
        ):
            ipsatize(data, [0, 5])

    def test_ipsatize_mixed_item_types(self):
        """Test that mixed string/int items raises TypeError."""
        data = pd.DataFrame({"item1": [1, 2], "item2": [2, 3]})
        with pytest.raises(
            TypeError, match="All items in 'items' must be either strings or integers"
        ):
            ipsatize(data, ["item1", 1])


class TestScore:
    """Tests for the score function."""

    def test_score_invalid_data_type(self):
        """Test that non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="Input 'data' must be a pandas DataFrame"):
            score([1, 2, 3], ["item1"], "csig")  # type: ignore[invalid-argument-type]

    def test_score_invalid_items_type(self):
        """Test that non-iterable items raises TypeError."""
        data = pd.DataFrame({"item1": [1, 2]})
        with pytest.raises(TypeError, match="Input 'items' must be a sequence"):
            score(data, "item1", "csig")

    def test_score_invalid_instrument_type(self):
        """Test that invalid instrument type raises TypeError."""
        data = pd.DataFrame({"item1": [1, 2]})
        with pytest.raises(
            TypeError, match="Input 'instrument' must be an Instrument instance"
        ):
            score(data, ["item1"], 123)  # type: ignore[invalid-argument-type]

    def test_score_delegates_to_instrument(self):
        """Test that score delegates to instrument.score method."""
        data = pd.DataFrame({"item1": [1, 2], "item2": [2, 3]})
        mock_instrument = Mock(spec=Instrument)
        mock_instrument.score.return_value = pd.DataFrame({"scale1": [1.5, 2.5]})

        result = score(
            data,
            ["item1", "item2"],
            mock_instrument,
            prefix="pre_",
            suffix="_post",
            na_rm=False,
            append=False,
        )

        mock_instrument.score.assert_called_once_with(
            data,
            ["item1", "item2"],
            prefix="pre_",
            suffix="_post",
            na_rm=False,
            append=False,
        )
        assert "scale1" in result.columns


class TestNormStandardize:
    """Tests for the norm_standardize function."""

    def test_norm_standardize_invalid_data_type(self):
        """Test that non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="Input 'data' must be a pandas DataFrame"):
            norm_standardize([1, 2, 3], ["scale1"], "csig", 1)  # type: ignore[invalid-argument-type]

    def test_norm_standardize_invalid_scales_type(self):
        """Test that non-iterable scales raises TypeError."""
        data = pd.DataFrame({"scale1": [1, 2]})
        with pytest.raises(TypeError, match="Input 'scales' must be a sequence"):
            norm_standardize(
                data,
                "csig",
                1,
                scales="scale1",
            )

    def test_norm_standardize_invalid_instrument_type(self):
        """Test that invalid instrument type raises TypeError."""
        data = pd.DataFrame({"scale1": [1, 2]})
        with pytest.raises(
            TypeError, match="Input 'instrument' must be an Instrument instance"
        ):
            norm_standardize(data, 123, 1, scales=["scale1"])  # type: ignore[invalid-argument-type]

    def test_norm_standardize_delegates_to_instrument(self):
        """Test norm_standardize delegates to instrument.norm_standardize method."""
        data = pd.DataFrame({"scale1": [1, 2], "scale2": [2, 3]})
        mock_instrument = Mock(spec=Instrument)
        mock_instrument.norm_standardize.return_value = pd.DataFrame(
            {
                "scale1_z": [-1.0, 1.0],
                "scale2_z": [-1.0, 1.0],
            }
        )

        result = norm_standardize(
            data,
            mock_instrument,
            1,
            scales=["scale1", "scale2"],
            prefix="pre_",
            suffix="_post",
            append=False,
        )

        mock_instrument.norm_standardize.assert_called_once_with(
            data,
            1,
            scales=["scale1", "scale2"],
            prefix="pre_",
            suffix="_post",
            append=False,
        )
        assert "scale1_z" in result.columns

    def test_norm_standardize_converts_sample_id_to_int(self):
        """Test that sample_id is converted to int."""
        data = pd.DataFrame({"scale1": [1, 2]})
        mock_instrument = Mock(spec=Instrument)
        mock_instrument.norm_standardize.return_value = pd.DataFrame(
            {"scale1_z": [-1.0, 1.0]}
        )

        norm_standardize(
            data,
            mock_instrument,
            "1",  # type: ignore[invalid-argument-type]
            scales=["scale1"],
        )

        # Check that int(sample_id) was passed
        call_args = mock_instrument.norm_standardize.call_args
        assert call_args[0][1] == 1
        assert isinstance(call_args[0][1], int)
