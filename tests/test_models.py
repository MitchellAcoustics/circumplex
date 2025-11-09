"""Unit tests for instrument data models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from circumplex.instruments.models import (
    _INSTRUMENTS,
    Instrument,
    InstrumentScale,
    NormativeSample,
    ResponseAnchor,
    ResponseItem,
    get_instrument,
    register_instrument,
    show_instruments,
)


# Fixtures
@pytest.fixture
def sample_scales() -> tuple[InstrumentScale, ...]:
    """Create sample instrument scales."""
    return (
        InstrumentScale(abbrev="PA", angle=90.0, items=(1, 2), label="Positive Affect"),
        InstrumentScale(
            abbrev="NA", angle=270.0, items=(3, 4), label="Negative Affect"
        ),
    )


@pytest.fixture
def sample_anchors() -> tuple[ResponseAnchor, ...]:
    """Create sample response anchors."""
    return (
        ResponseAnchor(value=1, label="Strongly Disagree"),
        ResponseAnchor(value=2, label="Disagree"),
        ResponseAnchor(value=3, label="Neutral"),
        ResponseAnchor(value=4, label="Agree"),
        ResponseAnchor(value=5, label="Strongly Agree"),
    )


@pytest.fixture
def sample_items() -> tuple[ResponseItem, ...]:
    """Create sample response items."""
    return (
        ResponseItem(item_id=1, text="I feel happy"),
        ResponseItem(item_id=2, text="I feel joyful"),
        ResponseItem(item_id=3, text="I feel sad"),
        ResponseItem(item_id=4, text="I feel anxious"),
    )


@pytest.fixture
def sample_norms() -> tuple[NormativeSample, ...]:
    """Create sample normative data."""
    stats = pd.DataFrame(
        {
            "scale": ["PA", "NA"],
            "mean": [3.5, 2.0],
            "sd": [0.8, 0.6],
        }
    )
    return (
        NormativeSample(
            sample_id=1,
            size=500,
            population="College Students",
            reference="Doe et al. (2020)",
            url="https://example.com/norms",
            statistics=stats,
        ),
    )


@pytest.fixture
def sample_instrument(
    sample_scales: tuple[InstrumentScale, ...],
    sample_anchors: tuple[ResponseAnchor, ...],
    sample_items: tuple[ResponseItem, ...],
    sample_norms: tuple[NormativeSample, ...],
) -> Instrument:
    """Create a sample instrument."""
    return Instrument(
        name="Test Affect Scale",
        abbrev="TAS",
        construct="Affect",
        reference="Test et al. (2023)",
        url="https://example.com/tas",
        status="open-access",
        scales=sample_scales,
        anchors=sample_anchors,
        items=sample_items,
        norms=sample_norms,
        prefix="TAS_",
        suffix="_score",
    )


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample item-level data for scoring."""
    return pd.DataFrame(
        {
            "item1": [5, 4, 3, 2, 1],
            "item2": [5, 4, 3, 2, 1],
            "item3": [1, 2, 3, 4, 5],
            "item4": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the global instrument registry before each test."""
    _INSTRUMENTS.clear()
    yield
    _INSTRUMENTS.clear()


# Test InstrumentScale
class TestInstrumentScale:
    """Tests for InstrumentScale dataclass."""

    def test_creation(self):
        """Test creating an InstrumentScale."""
        scale = InstrumentScale(
            abbrev="PA", angle=90.0, items=(1, 2, 3), label="Positive Affect"
        )
        assert scale.abbrev == "PA"
        assert scale.angle == 90.0
        assert scale.items == (1, 2, 3)
        assert scale.label == "Positive Affect"

    def test_immutable(self):
        """Test that InstrumentScale is immutable."""
        scale = InstrumentScale(abbrev="PA", angle=90.0, items=(1, 2), label="Test")
        with pytest.raises(AttributeError):
            scale.abbrev = "NA"  # type: ignore[invalid-assignment]


# Test ResponseAnchor
class TestResponseAnchor:
    """Tests for ResponseAnchor dataclass."""

    def test_creation(self):
        """Test creating a ResponseAnchor."""
        anchor = ResponseAnchor(value=1, label="Strongly Disagree")
        assert anchor.value == 1
        assert anchor.label == "Strongly Disagree"

    def test_immutable(self):
        """Test that ResponseAnchor is immutable."""
        anchor = ResponseAnchor(value=1, label="Test")
        with pytest.raises(AttributeError):
            anchor.value = 2  # type: ignore[invalid-assignment]


# Test ResponseItem
class TestResponseItem:
    """Tests for ResponseItem dataclass."""

    def test_creation(self):
        """Test creating a ResponseItem."""
        item = ResponseItem(item_id=1, text="I feel happy")
        assert item.item_id == 1
        assert item.text == "I feel happy"

    def test_immutable(self):
        """Test that ResponseItem is immutable."""
        item = ResponseItem(item_id=1, text="Test")
        with pytest.raises(AttributeError):
            item.text = "New text"  # type: ignore[invalid-assignment]


# Test NormativeSample
class TestNormativeSample:
    """Tests for NormativeSample dataclass."""

    def test_creation(self):
        """Test creating a NormativeSample."""
        stats = pd.DataFrame({"scale": ["PA"], "mean": [3.5], "sd": [0.8]})
        norm = NormativeSample(
            sample_id=1,
            size=100,
            population="Students",
            reference="Doe (2020)",
            url="https://example.com",
            statistics=stats,
        )
        assert norm.sample_id == 1
        assert norm.size == 100
        assert norm.population == "Students"
        assert isinstance(norm.statistics, pd.DataFrame)


# Test Instrument
class TestInstrument:
    """Tests for Instrument class."""

    def test_creation(self, sample_instrument: Instrument):
        """Test creating an Instrument."""
        assert sample_instrument.name == "Test Affect Scale"
        assert sample_instrument.abbrev == "TAS"
        assert sample_instrument.construct == "Affect"
        assert sample_instrument.n_scales == 2
        assert sample_instrument.n_items == 4

    def test_n_scales_property(self, sample_instrument: Instrument):
        """Test n_scales property."""
        assert sample_instrument.n_scales == 2

    def test_n_items_property(self, sample_instrument: Instrument):
        """Test n_items property."""
        assert sample_instrument.n_items == 4

    def test_n_items_no_items(
        self,
        sample_scales: tuple[InstrumentScale, ...],
        sample_anchors: tuple[ResponseAnchor, ...],
    ):
        """Test n_items when no items are provided."""
        inst = Instrument(
            name="Test",
            abbrev="TST",
            construct="Test",
            reference="Test",
            url="https://test.com",
            status="open-access",
            scales=sample_scales,
            anchors=sample_anchors,
            items=None,
        )
        assert inst.n_items == 0

    def test_scale_labels_property(self, sample_instrument: Instrument):
        """Test scale_labels property."""
        labels = sample_instrument.scale_labels
        assert labels == ["Positive Affect", "Negative Affect"]

    def test_scale_abbrevs_property(self, sample_instrument: Instrument):
        """Test scale_abbrevs property."""
        abbrevs = sample_instrument.scale_abbrevs
        assert abbrevs == ["PA", "NA"]

    def test_get_angles(self, sample_instrument: Instrument):
        """Test get_angles method."""
        angles = sample_instrument.get_angles()
        assert angles == [90.0, 270.0]

    def test_get_scale_success(self, sample_instrument: Instrument):
        """Test get_scale with valid abbreviation."""
        scale = sample_instrument.get_scale("PA")
        assert scale.abbrev == "PA"
        assert scale.label == "Positive Affect"

    def test_get_scale_failure(self, sample_instrument: Instrument):
        """Test get_scale with invalid abbreviation."""
        with pytest.raises(ValueError, match="Scale with abbreviation 'XX' not found"):
            sample_instrument.get_scale("XX")

    def test_get_item_success(self, sample_instrument: Instrument):
        """Test get_item with valid item ID."""
        item = sample_instrument.get_item(1)
        assert item.item_id == 1
        assert item.text == "I feel happy"

    def test_get_item_failure(self, sample_instrument: Instrument):
        """Test get_item with invalid item ID."""
        with pytest.raises(ValueError, match="Item with ID '99' not found"):
            sample_instrument.get_item(99)

    def test_get_item_no_items(
        self,
        sample_scales: tuple[InstrumentScale, ...],
        sample_anchors: tuple[ResponseAnchor, ...],
    ):
        """Test get_item when items are not available."""
        inst = Instrument(
            name="Test",
            abbrev="TST",
            construct="Test",
            reference="Test",
            url="https://test.com",
            status="open-access",
            scales=sample_scales,
            anchors=sample_anchors,
            items=None,
        )
        with pytest.raises(ValueError, match="does not have item text available"):
            inst.get_item(1)

    def test_repr(self, sample_instrument: Instrument):
        """Test __repr__ method."""
        repr_str = repr(sample_instrument)
        assert "TAS: Test Affect Scale" in repr_str
        assert "4 items, 2 scales, 1 normative data sets" in repr_str
        assert "Test et al. (2023)" in repr_str
        assert "https://example.com/tas" in repr_str

    def test_rich_repr(self, sample_instrument: Instrument):
        """Test __rich_repr__ method."""
        rich_repr = list(sample_instrument.__rich_repr__())
        assert len(rich_repr) == 4
        assert "TAS: Test Affect Scale" in rich_repr[0]

    def test_info_plain(self, sample_instrument: Instrument, capsys):
        """Test info method with plain output."""
        sample_instrument.info(rich_print=False)
        captured = capsys.readouterr()
        assert "TAS: Test Affect Scale" in captured.out
        assert "Positive Affect" in captured.out

    def test_info_scales_plain(self, sample_instrument: Instrument):
        """Test info_scales with plain output."""
        result = sample_instrument.info_scales(rich_print=False)
        assert isinstance(result, str)
        assert "The TAS contains 2 scales:" in result
        assert "PA: Positive Affect (90.0°)" in result
        assert "NA: Negative Affect (270.0°)" in result

    def test_info_scales_with_items_plain(self, sample_instrument: Instrument):
        """Test info_scales with items displayed."""
        result = sample_instrument.info_scales(items=True, rich_print=False)
        assert isinstance(result, str)
        assert "1. I feel happy" in result
        assert "3. I feel sad" in result

    def test_info_anchors_plain(self, sample_instrument: Instrument):
        """Test info_anchors with plain output."""
        result = sample_instrument.info_anchors(rich_print=False)
        assert isinstance(result, str)
        assert "5-point scale:" in result
        assert "1. Strongly Disagree" in result
        assert "5. Strongly Agree" in result

    def test_info_norms_plain(self, sample_instrument: Instrument):
        """Test info_norms with plain output."""
        result = sample_instrument.info_norms(rich_print=False)
        assert isinstance(result, str)
        assert "1 normative data set(s):" in result
        assert "500 College Students" in result
        assert "Doe et al. (2020)" in result

    def test_score_with_column_names(
        self, sample_instrument: Instrument, sample_data: pd.DataFrame
    ):
        """Test scoring with column names."""
        result = sample_instrument.score(
            sample_data,
            items=["item1", "item2", "item3", "item4"],
            prefix="",
            suffix="",
            append=False,
        )
        assert result.shape == (5, 2)
        assert "PA" in result.columns
        assert "NA" in result.columns
        # PA is mean of items 1, 2; NA is mean of items 3, 4
        assert result["PA"].iloc[0] == 5.0
        assert result["NA"].iloc[0] == 1.0

    def test_score_with_indices(
        self, sample_instrument: Instrument, sample_data: pd.DataFrame
    ):
        """Test scoring with column indices."""
        result = sample_instrument.score(
            sample_data,
            items=[0, 1, 2, 3],
            append=False,
        )
        assert result.shape == (5, 2)
        assert "PA" in result.columns
        assert "NA" in result.columns

    def test_score_with_prefix_suffix(
        self, sample_instrument: Instrument, sample_data: pd.DataFrame
    ):
        """Test scoring with prefix and suffix."""
        result = sample_instrument.score(
            sample_data,
            items=[0, 1, 2, 3],
            prefix="pre_",
            suffix="_suf",
            append=False,
        )
        assert "pre_PA_suf" in result.columns
        assert "pre_NA_suf" in result.columns

    def test_score_append(
        self, sample_instrument: Instrument, sample_data: pd.DataFrame
    ):
        """Test scoring with append=True."""
        result = sample_instrument.score(
            sample_data,
            items=[0, 1, 2, 3],
            append=True,
        )
        assert result.shape == (5, 6)  # 4 original + 2 scores
        assert "item1" in result.columns
        assert "PA" in result.columns

    def test_score_with_missing_data(self, sample_instrument: Instrument):
        """Test scoring with missing values."""
        data = pd.DataFrame(
            {
                "item1": [5, 4, np.nan, 2, 1],
                "item2": [5, 4, 3, np.nan, 1],
                "item3": [1, 2, 3, 4, 5],
                "item4": [1, 2, 3, 4, 5],
            }
        )
        result = sample_instrument.score(
            data, items=[0, 1, 2, 3], na_rm=True, append=False
        )
        assert not np.isnan(result["PA"].iloc[2])  # Should compute from remaining item
        assert not np.isnan(result["PA"].iloc[3])

    def test_score_mixed_item_types_error(
        self, sample_instrument: Instrument, sample_data: pd.DataFrame
    ):
        """Test scoring with mixed item types raises error."""
        with pytest.raises(TypeError, match="must be either strings or integers"):
            sample_instrument.score(sample_data, items=["item1", 1])

    def test_norm_standardize_success(self, sample_instrument: Instrument):
        """Test norm_standardize with valid sample ID."""
        data = pd.DataFrame(
            {
                "PA": [4.3, 3.5, 2.7],
                "NA": [2.6, 2.0, 1.4],
            }
        )
        result = sample_instrument.norm_standardize(
            data,
            scales=["PA", "NA"],
            sample_id=1,
            append=False,
        )
        assert "PA_z" in result.columns
        assert "NA_z" in result.columns
        # Check standardization: (4.3 - 3.5) / 0.8 = 1.0
        assert np.isclose(result["PA_z"].iloc[0], 1.0)
        # Check standardization: (3.5 - 3.5) / 0.8 = 0.0
        assert np.isclose(result["PA_z"].iloc[1], 0.0)

    def test_norm_standardize_with_indices(self, sample_instrument: Instrument):
        """Test norm_standardize with scale indices."""
        data = pd.DataFrame(
            {
                "PA": [4.3, 3.5, 2.7],
                "NA": [2.6, 2.0, 1.4],
            }
        )
        result = sample_instrument.norm_standardize(
            data,
            scales=[0, 1],  # Use indices instead of names
            sample_id=1,
            append=False,
        )
        assert "PA_z" in result.columns
        assert "NA_z" in result.columns

    def test_norm_standardize_append(self, sample_instrument: Instrument):
        """Test norm_standardize with append=True."""
        data = pd.DataFrame(
            {
                "PA": [4.3, 3.5, 2.7],
                "NA": [2.6, 2.0, 1.4],
            }
        )
        result = sample_instrument.norm_standardize(
            data,
            scales=["PA", "NA"],
            sample_id=1,
            append=True,
        )
        assert "PA" in result.columns
        assert "PA_z" in result.columns

    def test_norm_standardize_custom_affix(self, sample_instrument: Instrument):
        """Test norm_standardize with custom prefix and suffix."""
        data = pd.DataFrame(
            {
                "PA": [4.3],
                "NA": [2.6],
            }
        )
        result = sample_instrument.norm_standardize(
            data,
            scales=["PA", "NA"],
            sample_id=1,
            prefix="std_",
            suffix="_score",
            append=False,
        )
        assert "std_PA_score" in result.columns
        assert "std_NA_score" in result.columns

    def test_norm_standardize_invalid_sample_id(self, sample_instrument: Instrument):
        """Test norm_standardize with invalid sample ID."""
        data = pd.DataFrame({"PA": [4.3], "NA": [2.6]})
        with pytest.raises(
            ValueError, match="Normative sample with ID '999' not found"
        ):
            sample_instrument.norm_standardize(data, scales=["PA"], sample_id=999)


# Test Registry Functions
class TestRegistryFunctions:
    """Tests for global registry functions."""

    def test_register_instrument(self, sample_instrument: Instrument):
        """Test registering an instrument."""
        register_instrument("TAS", sample_instrument)
        assert "tas" in _INSTRUMENTS
        assert _INSTRUMENTS["tas"] == sample_instrument

    def test_get_instrument_success(self, sample_instrument: Instrument):
        """Test retrieving a registered instrument."""
        register_instrument("TAS", sample_instrument)
        retrieved = get_instrument("TAS")
        assert retrieved == sample_instrument

    def test_get_instrument_case_insensitive(self, sample_instrument: Instrument):
        """Test that get_instrument is case-insensitive."""
        register_instrument("TAS", sample_instrument)
        retrieved = get_instrument("tas")
        assert retrieved == sample_instrument

    def test_get_instrument_not_found(self):
        """Test get_instrument with unregistered instrument."""
        with pytest.raises(ValueError, match="Instrument 'XYZ' not found"):
            get_instrument("XYZ")

    def test_show_instruments_empty(self, capsys):
        """Test show_instruments with empty registry."""
        show_instruments(rich_print=False)
        captured = capsys.readouterr()
        assert "0 instruments" in captured.out

    def test_show_instruments_with_instruments(
        self, sample_instrument: Instrument, capsys
    ):
        """Test show_instruments with registered instruments."""
        register_instrument("TAS", sample_instrument)
        show_instruments(rich_print=False)
        captured = capsys.readouterr()
        assert "1 instruments" in captured.out
        assert "TAS" in captured.out
        assert "Test Affect Scale" in captured.out
