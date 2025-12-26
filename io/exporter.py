from __future__ import annotations

import pandas as pd


def to_csv(df: pd.DataFrame, path: str) -> None:
    
    """
    Export a Pandas DataFrame to a CSV file.

    The DataFrame is saved without the index to ensure clean,
    portable output suitable for further analysis or reporting.
    """
    # Save DataFrame to CSV file without index column
    df.to_csv(path, index=False)
