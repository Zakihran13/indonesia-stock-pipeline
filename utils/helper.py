from typing import List, Any
import pandas as pd
import inflection
import re

def split_batch(data: List[Any], batch_size: int) -> List[List[Any]]:
    """Splits a list into smaller lists of the specified size."""
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0.")

    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def _to_snake_case(col: Any) -> str:
    col_str = str(col).strip()
    col_str = col_str.replace("&", "_and_")
    col_str = col_str.replace("%", "_percent")
    col_str = col_str.replace("#", "_num_")
    col_str = inflection.underscore(col_str)
    col_str = re.sub(r"[^a-zA-Z0-9]+", "_", col_str)
    col_str = re.sub(r"_+", "_", col_str)
    return col_str.strip("_").lower()


def snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames all columns in a pandas DataFrame to snake_case
    using the inflection library and regex formatting.
    """
    df = df.copy()
    df.columns = [_to_snake_case(col) for col in df.columns]
    return df
