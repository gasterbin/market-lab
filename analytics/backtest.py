from __future__ import annotations

import pandas as pd


def ema_cross_backtest(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> dict:
    data = df.copy()

    data["ema_fast"] = data["close"].ewm(span=fast, adjust=False).mean()
    data["ema_slow"] = data["close"].ewm(span=slow, adjust=False).mean()

    data["signal"] = (data["ema_fast"] > data["ema_slow"]).astype(int)  # 1 long, 0 flat
    data["position"] = data["signal"].shift(1).fillna(0)

    data["return"] = data["close"].pct_change().fillna(0)
    data["strategy_return"] = data["position"] * data["return"]
    data["equity"] = (1 + data["strategy_return"]).cumprod()

    total_return = float(data["equity"].iloc[-1] - 1)

    peak = data["equity"].cummax()
    drawdown = (data["equity"] / peak) - 1
    max_drawdown = float(drawdown.min())

    trades = int((data["position"].diff().abs() > 0).sum())

    return {
        "fast": fast,
        "slow": slow,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "trades": trades,
    }
