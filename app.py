from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class BinanceKlinesRequest:
    """
    Parameters for a Binance klines (candlestick) request.

    Times are expected in milliseconds (ms) since Unix epoch, as required by the API.
    """
    symbol: str
    interval: str = "1h"
    limit: int = 200
    start_time_ms: int | None = None
    end_time_ms: int | None = None


class BinanceClient:
    """
    Minimal client for Binance public REST API (no API key needed).

    Uses the Klines/Candlestick endpoint:
    https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
    """

    BASE_URL = "https://api.binance.com"

    def get_klines(self, req: BinanceKlinesRequest) -> list[list[Any]]:
        """
        Fetch OHLCV klines from Binance.

        Returns:
            A list of klines (each kline is a list of values in Binance format).
        """
        url = f"{self.BASE_URL}/api/v3/klines"

        # Build query parameters according to Binance API spec
        params: dict[str, Any] = {
            "symbol": req.symbol.upper(),
            "interval": req.interval,
            "limit": int(req.limit),
        }

        # Optional time filters (milliseconds since epoch)
        if req.start_time_ms is not None:
            params["startTime"] = int(req.start_time_ms)
        if req.end_time_ms is not None:
            params["endTime"] = int(req.end_time_ms)

        # Perform network request with a timeout to avoid hanging forever
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()

        data = response.json()

        # Basic validation: Binance should return a list of klines
        if not isinstance(data, list):
            raise ValueError("Unexpected response format from Binance API")

        return data
