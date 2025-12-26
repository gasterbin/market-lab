from __future__ import annotations

import pandas as pd


def ema_cross_backtest(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
) -> dict:
    """
    Run a simple EMA crossover backtest on OHLCV market data.

    The strategy goes long when the fast EMA is above the slow EMA
    and stays flat otherwise. Returns basic performance metrics.
    """
    # Work on a copy to avoid modifying the original DataFrame
    data = df.copy()

    # Calculate exponential moving averages
    data["ema_fast"] = data["close"].ewm(span=fast, adjust=False).mean()
    data["ema_slow"] = data["close"].ewm(span=slow, adjust=False).mean()

    # Trading signal: 1 = long position, 0 = flat
    data["signal"] = (data["ema_fast"] > data["ema_slow"]).astype(int)

    # Shift signal to simulate entering the position on the next candle
    data["position"] = data["signal"].shift(1).fillna(0)

    # Market returns based on close price
    data["return"] = data["close"].pct_change().fillna(0)

    # Strategy returns depend on the current position
    data["strategy_return"] = data["position"] * data["return"]

    # Equity curve assuming reinvestment of profits
    data["equity"] = (1 + data["strategy_return"]).cumprod()

    # Total strategy return
    total_return = float(data["equity"].iloc[-1] - 1)

    # Drawdown calculation
    peak = data["equity"].cummax()
    drawdown = (data["equity"] / peak) - 1
    max_drawdown = float(drawdown.min())

    # Count number of trades (position changes)
    trades = int((data["position"].diff().abs() > 0).sum())

    return {
        "fast": fast,
        "slow": slow,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "trades": trades,
    }
