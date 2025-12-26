from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from services.binance import BinanceClient, BinanceKlinesRequest


BINANCE_COLUMNS = [
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


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _klines_to_df(klines: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(klines, columns=BINANCE_COLUMNS)

    # Convert types
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Keep only what we need for analysis/export
    return df[["open_time", "open", "high", "low", "close", "volume"]].copy()


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Support both saved formats: with open_time as string or already parsed
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    return df


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_parent_dir(path)
    df.to_csv(path, index=False)


def ema(series: pd.Series, span: int) -> pd.Series:
    # Pandas EWM is standard and stable
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    # Simple RSI implementation with numpy/pandas
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain_s = pd.Series(gain, index=close.index).rolling(period).mean()
    loss_s = pd.Series(loss, index=close.index).rolling(period).mean()

    rs = gain_s / (loss_s.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out


def cmd_fetch(args: argparse.Namespace) -> int:
    client = BinanceClient()
    req = BinanceKlinesRequest(
        symbol=args.ticker,
        interval=args.interval,
        limit=args.limit,
        start_time_ms=args.start_ms,
        end_time_ms=args.end_ms,
    )

    klines = client.get_klines(req)
    df = _klines_to_df(klines)

    if args.out:
        out_path = Path(args.out)
        _write_csv(df, out_path)
        print(f"Saved: {out_path.resolve()}")
    else:
        # Print head to stdout if no output file
        print(df.head(10).to_string(index=False))

    return 0


def cmd_indicators(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    df = _read_csv(in_path)

    if "close" not in df.columns:
        raise ValueError("Input CSV must contain 'close' column.")

    df[f"ema_{args.ema_fast}"] = ema(df["close"], args.ema_fast)
    df[f"ema_{args.ema_slow}"] = ema(df["close"], args.ema_slow)

    if args.with_rsi:
        df[f"rsi_{args.rsi_period}"] = rsi(df["close"], args.rsi_period)

    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_indicators.csv")
    _write_csv(df, out_path)
    print(f"Saved: {out_path.resolve()}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    df = _read_csv(in_path)

    for col in ["close"]:
        if col not in df.columns:
            raise ValueError("Input CSV must contain 'close' column.")

    fast = ema(df["close"], args.fast)
    slow = ema(df["close"], args.slow)

    # Signal: 1 when fast > slow, else 0
    signal = (fast > slow).astype(int)

    # Positions change => trades
    positions = signal.diff().fillna(0)

    # Returns
    ret = df["close"].pct_change().fillna(0)

    # Strategy returns: hold position from previous candle (avoid lookahead)
    strat_ret = ret * signal.shift(1).fillna(0)

    equity = (1 + strat_ret).cumprod()

    # Summary
    total_return = float(equity.iloc[-1] - 1)
    trades = int((positions.abs() == 1).sum())
    win_rate = float((strat_ret[strat_ret != 0] > 0).mean()) if (strat_ret != 0).any() else 0.0

    out_df = df.copy()
    out_df[f"ema_{args.fast}"] = fast
    out_df[f"ema_{args.slow}"] = slow
    out_df["signal"] = signal
    out_df["strategy_return"] = strat_ret
    out_df["equity"] = equity

    if args.out:
        out_path = Path(args.out)
        _write_csv(out_df, out_path)
        print(f"Saved: {out_path.resolve()}")

    print("Backtest summary")
    print(f"  fast EMA: {args.fast}")
    print(f"  slow EMA: {args.slow}")
    print(f"  trades:   {trades}")
    print(f"  win_rate: {win_rate:.2%}")
    print(f"  return:   {total_return:.2%}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="market-lab",
        description="Market Lab CLI: fetch OHLCV from Binance, compute indicators, run EMA crossover backtest.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # fetch
    f = sub.add_parser("fetch", help="Fetch OHLCV klines from Binance")
    f.add_argument("--ticker", required=True, help="Symbol, e.g. BTCUSDT")
    f.add_argument("--interval", default="1h", help="Kline interval, e.g. 1m, 5m, 1h, 1d")
    f.add_argument("--limit", type=int, default=200, help="Number of candles (max depends on API)")
    f.add_argument("--start-ms", type=int, default=None, help="Start time in ms since epoch (optional)")
    f.add_argument("--end-ms", type=int, default=None, help="End time in ms since epoch (optional)")
    f.add_argument("--out", default=None, help="Output CSV path (optional)")
    f.set_defaults(func=cmd_fetch)

    # indicators
    ind = sub.add_parser("indicators", help="Compute indicators (EMA, optional RSI) on a CSV")
    ind.add_argument("--input", required=True, help="Input CSV path")
    ind.add_argument("--ema-fast", type=int, default=12, help="Fast EMA span")
    ind.add_argument("--ema-slow", type=int, default=26, help="Slow EMA span")
    ind.add_argument("--with-rsi", action="store_true", help="Add RSI column")
    ind.add_argument("--rsi-period", type=int, default=14, help="RSI period")
    ind.add_argument("--out", default=None, help="Output CSV path (optional)")
    ind.set_defaults(func=cmd_indicators)

    # backtest
    bt = sub.add_parser("backtest", help="Run EMA crossover backtest on a CSV")
    bt.add_argument("--input", required=True, help="Input CSV path (must contain close)")
    bt.add_argument("--fast", type=int, default=12, help="Fast EMA span")
    bt.add_argument("--slow", type=int, default=26, help="Slow EMA span")
    bt.add_argument("--out", default=None, help="Output CSV with signals/equity (optional)")
    bt.set_defaults(func=cmd_backtest)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
