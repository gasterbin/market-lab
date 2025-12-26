from __future__ import annotations

import argparse
import json
import sys

from analytics.backtest import ema_cross_backtest
from analytics.indicators import apply_indicators
from io.exporter import to_csv
from io.loader import klines_to_df, read_csv
from services.binance import BinanceClient, BinanceKlinesRequest


def cmd_fetch(args: argparse.Namespace) -> int:
    client = BinanceClient()
    req = BinanceKlinesRequest(
        symbol=args.ticker,
        interval=args.interval,
        limit=args.limit,
    )
    klines = client.get_klines(req)
    df = klines_to_df(klines)

    if args.out:
        to_csv(df, args.out)
        print(f"Saved {len(df)} rows to {args.out}")
    else:
        print(df.tail(5).to_string(index=False))

    return 0


def cmd_indicators(args: argparse.Namespace) -> int:
    df = read_csv(args.input)

    out = apply_indicators(
        df,
        sma=args.sma,
        ema=args.ema,
        rsi=args.rsi,
        macd=not args.no_macd,
        bb=not args.no_bb,
        vol_window=args.vol,
    )

    if args.out:
        to_csv(out, args.out)
        print(f"Saved {len(out)} rows with indicators to {args.out}")
    else:
        print(out.tail(10).to_string(index=False))

    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    df = read_csv(args.input)
    result = ema_cross_backtest(df, fast=args.fast, slow=args.slow)
    print(json.dumps(result, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    df = read_csv(args.input)

    last_price = float(df["close"].iloc[-1])
    returns = df["close"].pct_change().dropna()

    summary = {
        "rows": int(len(df)),
        "last_price": last_price,
        "mean_return": float(returns.mean()) if len(returns) else None,
        "volatility_std": float(returns.std()) if len(returns) else None,
        "best_return": float(returns.max()) if len(returns) else None,
        "worst_return": float(returns.min()) if len(returns) else None,
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(json.dumps(summary, indent=2))
        print(f"Saved report to {args.out}")
    else:
        print(json.dumps(summary, indent=2))

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="market-lab",
        description="CLI app to fetch Binance market data, compute indicators, and run a simple backtest.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("fetch", help="Fetch OHLCV candles from Binance")
    pf.add_argument("--ticker", required=True, help="Symbol like BTCUSDT, ETHUSDT")
    pf.add_argument("--interval", default="1h", help="Interval: 1m, 5m, 1h, 1d, etc.")
    pf.add_argument("--limit", type=int, default=200, help="Number of candles (max 1000)")
    pf.add_argument("--out", default="data.csv", help="Output CSV path")
    pf.set_defaults(func=cmd_fetch)

    pi = sub.add_parser("indicators", help="Compute indicators using pandas/numpy")
    pi.add_argument("--input", required=True, help="Input CSV (from fetch)")
    pi.add_argument("--out", default="data_ind.csv", help="Output CSV path")
    pi.add_argument("--sma", type=int, default=20, help="SMA window")
    pi.add_argument("--ema", type=int, default=20, help="EMA span")
    pi.add_argument("--rsi", type=int, default=14, help="RSI period")
    pi.add_argument("--vol", type=int, default=20, help="Volatility window (std of returns)")
    pi.add_argument("--no-macd", action="store_true", help="Disable MACD")
    pi.add_argument("--no-bb", action="store_true", help="Disable Bollinger Bands")
    pi.set_defaults(func=cmd_indicators)

    pb = sub.add_parser("backtest", help="Run EMA crossover backtest")
    pb.add_argument("--input", required=True, help="Input CSV (from fetch)")
    pb.add_argument("--fast", type=int, default=12, help="Fast EMA span")
    pb.add_argument("--slow", type=int, default=26, help="Slow EMA span")
    pb.set_defaults(func=cmd_backtest)

    pr = sub.add_parser("report", help="Quick statistical report")
    pr.add_argument("--input", required=True, help="Input CSV (from fetch)")
    pr.add_argument("--out", default="", help="Optional output file (json)")
    pr.set_defaults(func=cmd_report)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
