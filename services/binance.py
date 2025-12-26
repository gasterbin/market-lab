from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class BinanceKlinesRequest:
    symbol: str
    interval: str = "1h"
    limit: int = 200
    start_time_ms: int | None = None
    end_time_ms: int | None = None


class BinanceClient:
    """
    Minimal client for Binance public REST API (no API key needed).
    Docs: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
    """

    BASE_URL = "https://api.binance.com"

    def get_klines(self, req: BinanceKlinesRequest) -> list[list[Any]]:
        url = f"{self.BASE_URL}/api/v3/klines"
        params: dict[str, Any] = {
            "symbol": req.symbol.upper(),
            "interval": req.interval,
            "limit": int(req.limit),
        }
        if req.start_time_ms is not None:
            params["startTime"] = int(req.start_time_ms)
        if req.end_time_ms is not None:
            params["endTime"] = int(req.end_time_ms)

        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        if not isinstance(data, list):
            raise ValueError("Unexpected response format from Binance API")

        return data
