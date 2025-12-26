from __future__ import annotations

from typing import Any

import pandas as pd


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
    df = pd.DataFrame(klines, columns=KLINE_COLUMNS)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

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
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["number_of_trades"] = (
        pd.to_numeric(df["number_of_trades"], errors="coerce").fillna(0).astype(int)
    )

    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["open_time", "close_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df
