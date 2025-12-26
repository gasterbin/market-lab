from __future__ import annotations

import pandas as pd


def to_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
