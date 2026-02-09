# yfinance Stock Downloader

A Python tool for bulk-downloading historical stock price data from Yahoo Finance. Point it at a directory of CSV files containing ticker symbols and it auto-discovers them, downloads OHLCV data for each ticker, and organizes the output by source.

## Usage

```bash
# Auto-discovery: scans data/ for CSVs, detects ticker columns, downloads everything
python main.py

# JSON configuration for complex setups
python main.py --config config.json

# Process specific sources only
python main.py --sources etf_list,iwb_holdings

# Custom date range and rate limiting
python main.py --start-date 2020-01-01 --chunk-size 10 --sleep 3

# Legacy mode (original hardcoded behavior)
python main.py --legacy
```

### Auto-discovery

The default mode scans the `data/` directory for CSV files and looks for columns named `ticker`, `symbol`, or common variants. Each CSV becomes a "source" with its own output directory under `output/`.

### JSON configuration

For more control, create a `config.json` specifying sources, ticker columns, filters, and output paths:

```bash
python main.py --create-config   # generates a sample config.json
python main.py --config config.json
```

## Features

- **Resume capability**: Skips tickers that already have output files
- **Rate limiting**: Configurable chunk sizes and sleep intervals to stay within Yahoo Finance limits
- **Retry with backoff**: Automatic retries on network failures
- **Validation**: Ticker symbol validation and minimum data point checks
- **Progress tracking**: tqdm progress bars with per-chunk status

## Output

```
output/
├── etf_list_data/
│   ├── VOO.csv
│   ├── SPY.csv
│   └── ...
├── iwb_holdings_data/
│   ├── AAPL.csv
│   ├── MSFT.csv
│   └── ...
└── logs/
    └── stock_download.log
```

Each output CSV contains daily OHLCV data from Yahoo Finance.

## Installation

```bash
git clone https://github.com/NashC/yfinance-stock-downloader.git
cd yfinance-stock-downloader
uv venv && source .venv/bin/activate
uv pip install -e .
```

Requires Python 3.9+. Core dependencies: pandas, yfinance, tqdm.

## Testing

```bash
pytest
```

## License

MIT
