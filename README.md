# Market Lab CLI

## Project description
Market Lab CLI is a simple Python command-line application for working with market data.
The project fetches OHLCV candlestick data from the Binance public API, processes it using Pandas and NumPy, and allows basic technical analysis and a simple EMA crossover backtest.

---

## How to run locally

### 1. Create and activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python app.py --help
```

## How to run via Docker

### 1. Build Docker image
```bash
docker build -t market-lab:1 .
```

### 2. Run container
```bash
docker run --rm market-lab:1 --help
```

# To save output files to your local folder:
```bash
docker run --rm -v "$PWD:/data" market-lab:1 fetch --ticker BTCUSDT --interval 1h --limit 200 --out /data/btc.csv
```

## Example CLI commands

### Fetch market data:
```bash
python app.py fetch --ticker BTCUSDT --interval 1h --limit 200 --out btc.csv
```

### Compute technical indicators:
```bash
python app.py indicators --input btc.csv --out btc_ind.csv
```

### Run EMA crossover backtest:
```bash
python app.py backtest --input btc.csv --fast 12 --slow 26
```

### Generate a simple report:
```bash
python app.py report --input btc.csv
```


## API used

Binance Spot API (Klines / Candlesticks)

Documentation: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data