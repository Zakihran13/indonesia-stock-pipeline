from typing import List, Any
import pandas as pd
import inflection


def split_batch(data: List[Any], batch_size: int) -> List[List[Any]]:
    """Splits a list into smaller lists of the specified size."""
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0.")

    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames all columns in a pandas DataFrame to snake_case
    using the inflection library.
    """
    df = df.copy()
    df.columns = [inflection.underscore(col) for col in df.columns]
    return df
