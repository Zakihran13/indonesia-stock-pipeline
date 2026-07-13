from typing import List, Any


def split_batch(data: List[Any], batch_size: int) -> List[List[Any]]:
    """Splits a list into smaller lists of the specified size."""
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0.")
        
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]