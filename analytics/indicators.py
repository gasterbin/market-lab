from __future__ import annotations

import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return"] = out["close"].pct_change()
    out["cum_return"] = (1 + out["return"].fillna(0)).cumprod() - 1
    return out


def add_sma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    out[f"sma_{window}"] = out["close"].rolling(window).mean()
    return out


def add_ema(df: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    out = df.copy()
    out[f"ema_{span}"] = out["close"].ewm(span=span, adjust=False).mean()
    return out


def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    out[f"vol_{window}"] = out["close"].pct_change().rolling(window).std()
    return out


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    out = df.copy()

    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return out


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    out = df.copy()
    ema_fast = out["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["close"].ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()

    out[f"macd_{fast}_{slow}"] = macd
    out[f"macd_signal_{signal}"] = macd_signal
    out["macd_hist"] = macd - macd_signal
    return out


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    out = df.copy()
    ma = out["close"].rolling(window).mean()
    sd = out["close"].rolling(window).std()

    out[f"bb_mid_{window}"] = ma
    out[f"bb_up_{window}"] = ma + k * sd
    out[f"bb_low_{window}"] = ma - k * sd
    return out


def apply_indicators(
    df: pd.DataFrame,
    sma: int | None = 20,
    ema: int | None = 20,
    rsi: int | None = 14,
    macd: bool = True,
    bb: bool = True,
    vol_window: int | None = 20,
) -> pd.DataFrame:
    out = df.copy()
    out = add_returns(out)

    if sma is not None:
        out = add_sma(out, sma)
    if ema is not None:
        out = add_ema(out, ema)
    if rsi is not None:
        out = add_rsi(out, rsi)
    if macd:
        out = add_macd(out)
    if bb:
        out = add_bollinger_bands(out)
    if vol_window is not None:
        out = add_volatility(out, vol_window)

    return out
