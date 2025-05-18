import pandas as pd


def validate_dataframe(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Validates a Pandas DataFrame for required columns and non-emptiness.

    Args:
        df: The Pandas DataFrame to validate.
        required_columns: A list of column names that must be present in the DataFrame.

    Raises:
        ValueError: If the DataFrame is empty or if any required columns are missing.
    """
    if df.empty:
        raise ValueError("DataFrame is empty. It must contain at least one row.")

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"DataFrame is missing required columns: {', '.join(missing_columns)}"
        )

    return None
