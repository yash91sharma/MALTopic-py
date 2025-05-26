import pandas as pd
import pytest

from src.maltopic.utils import validate_dataframe


class TestValidateDataFrame:
    # Test that a valid DataFrame with all required columns passes validation
    def test_valid_dataframe_with_required_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        # Should not raise for subset of columns
        assert validate_dataframe(df, ["a", "b"]) is None
        # Should not raise for all columns present
        assert validate_dataframe(df, ["a", "b", "c"]) is None

    # Test that missing required columns raises ValueError and lists missing columns
    @pytest.mark.parametrize(
        "df, required, missing",
        [
            (pd.DataFrame({"x": [1, 2], "y": [3, 4]}), ["x", "y", "z"], ["z"]),
            (pd.DataFrame({"x": [1, 2], "y": [3, 4]}), ["y", "foo"], ["foo"]),
            (pd.DataFrame({"foo": [1, 2]}), ["bar", "baz"], ["bar", "baz"]),
        ],
    )
    def test_missing_required_columns(self, df, required, missing):
        # Should raise ValueError if any required columns are missing
        with pytest.raises(ValueError) as excinfo:
            validate_dataframe(df, required)
        msg = str(excinfo.value)
        assert "missing required columns" in msg
        # Each missing column should be mentioned in the error message
        for col in missing:
            assert col in msg

    # Test that an empty DataFrame raises ValueError
    def test_empty_dataframe_raises(self):
        df = pd.DataFrame(columns=["a", "b"])
        # Should raise ValueError for empty DataFrame
        with pytest.raises(ValueError) as excinfo:
            validate_dataframe(df, ["a"])
        assert "DataFrame is empty" in str(excinfo.value)

    # Test that no required columns means always valid if DataFrame is not empty
    def test_no_required_columns(self):
        df = pd.DataFrame({"a": [1]})
        # Should not raise if required_columns is empty
        assert validate_dataframe(df, []) is None
