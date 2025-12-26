from __future__ import annotations

from typing import Any

import pandas as pd


# Column names returned by Binance Klines (candlestick) API
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


def klines_to_df(klines: list[list[Any]]) -> pd.DataFrame:
    """
    Convert raw Binance kline data to a clean Pandas DataFrame.

    This function:
    - assigns meaningful column names
    - converts timestamps to UTC datetime
    - casts numeric columns to appropriate types
    - sorts data by open time
    """
    # Create DataFrame from raw API response
    df = pd.DataFrame(klines, columns=KLINE_COLUMNS)

    # Convert timestamps from milliseconds to UTC datetime
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Columns that should be numeric for analysis
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]

    # Convert numeric columns, coercing invalid values to NaN
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Number of trades must be integer
    df["number_of_trades"] = (
        pd.to_numeric(df["number_of_trades"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # Ensure data is ordered chronologically
    df = df.sort_values("open_time").reset_index(drop=True)

    return df


def read_csv(path: str) -> pd.DataFrame:
    """
    Read market data from a CSV file and parse datetime columns.

    This function ensures consistency between data loaded
    from the Binance API and data loaded from disk.
    """
    df = pd.read_csv(path)

    # Parse datetime columns if they exist
    for col in ["open_time", "close_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
