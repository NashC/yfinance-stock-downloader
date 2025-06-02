# Enhanced Multi-Source Stock Price Data Downloader

A production-ready Python application for downloading historical stock price data from Yahoo Finance with support for multiple data sources, robust error handling, and intelligent rate limiting.

## 🚀 Features

- **Multi-source data support** - Individual stocks (Russell 1000) + ETF universe
- **Modular architecture** - Clean separation of concerns with organized packages
- **Intelligent rate limiting** - Respects Yahoo Finance API limits with exponential backoff
- **Progress tracking** - Real-time progress bars and comprehensive logging
- **Resume capability** - Automatically skips already downloaded files
- **Data validation** - Validates ticker symbols and data quality
- **Configurable processing** - Multiple processing modes (stocks, ETFs, both, auto)
- **Production-ready** - Type hints, error handling, and comprehensive testing

## 📁 Project Structure

```
yfinance_stock_prices/
├── src/                          # Main package
│   ├── __init__.py              # Package exports
│   ├── config.py                # Configuration classes and enums
│   ├── validators.py            # Ticker and data validation
│   └── downloader.py            # Main downloader class
├── tests/                       # Test suite
│   ├── __init__.py
│   └── test_downloader.py       # Test version with limited tickers
├── data/                        # Input data sources
│   ├── IWB_holdings_250529.csv # Russell 1000 stock holdings
│   └── etf_list.csv            # ETF universe data
├── stock_data/                  # Output: Individual stock CSV files
├── etf_data/                    # Output: ETF CSV files
├── main.py                      # Main entry point
├── pyproject.toml              # Modern Python packaging
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🛠 Installation & Setup

### Modern Setup with uv (Recommended)

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install project with dependencies
uv pip install -e .
```

### Alternative Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Usage

### Command-Line Tool (Recommended)

```bash
# Run the downloader
download-stock-data
```

### Direct Python Execution

```bash
# Run main script
python main.py

# Run test version (limited tickers)
python tests/test_downloader.py
```

### Programmatic Usage

```python
from src import StockDataDownloader, DownloadConfig, ProcessingMode

# Configure downloader
config = DownloadConfig(
    start_date="2020-01-01",
    processing_mode=ProcessingMode.BOTH,
    chunk_size=25,
    sleep_between_chunks=5
)

# Run downloader
downloader = StockDataDownloader(config)
downloader.run()
```

## ⚙️ Configuration

### Processing Modes

- `ProcessingMode.BOTH` - Download both stocks and ETFs (default)
- `ProcessingMode.STOCKS` - Download only individual stocks
- `ProcessingMode.ETFS` - Download only ETFs
- `ProcessingMode.AUTO` - Auto-detect available sources

### Rate Limiting Settings

```python
config = DownloadConfig(
    chunk_size=25,              # Tickers per batch
    sleep_between_chunks=5,     # Seconds between batches (~12 req/min)
    individual_sleep=2,         # Seconds between individual retries (~30 req/min)
    max_retries=3,              # Maximum retry attempts
    initial_backoff=2           # Exponential backoff (2→4→8 seconds)
)
```

### Data Sources

The application supports multiple CSV data sources:

```python
# Individual Stocks (Russell 1000)
"iwb_holdings": {
    "file": "data/IWB_holdings_250529.csv",
    "ticker_column": "Ticker",
    "output_dir": "stock_data"
}

# ETF Universe
"etf_list": {
    "file": "data/etf_list.csv", 
    "ticker_column": "Symbol",
    "output_dir": "etf_data"
}
```

## 📊 Data Output

Each ticker generates a CSV file with OHLCV data:

```csv
Date,Open,High,Low,Close,Adj Close,Volume
2020-01-02,74.0600,75.1500,73.7975,75.0875,73.1234,135480400
2020-01-03,74.2875,75.1450,74.1250,74.3575,72.4119,146322800
```

## 🧪 Testing

### Quick Test (5 tickers per source)

```bash
python tests/test_downloader.py
```

### Development Testing

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run linting and formatting
black src/ tests/
isort src/ tests/
mypy src/
ruff check src/ tests/
```

## 📈 Performance & Rate Limiting

The application implements conservative rate limiting to respect Yahoo Finance API limits:

- **Chunk downloads**: ~12 requests/minute (well below 60/min limit)
- **Individual retries**: ~30 requests/minute (safe buffer)
- **Total theoretical max**: ~42 requests/minute (30% buffer)
- **Exponential backoff**: Automatic retry with increasing delays

## 🔧 Architecture

### Modular Design

- **`src/config.py`** - Configuration classes and enums
- **`src/validators.py`** - Ticker validation and data cleaning
- **`src/downloader.py`** - Main download logic and orchestration
- **`main.py`** - Entry point and CLI interface

### Key Classes

- **`DownloadConfig`** - Main configuration dataclass
- **`DataSourceConfig`** - Individual data source configuration
- **`TickerValidator`** - Ticker symbol validation and cleaning
- **`DataValidator`** - Downloaded data validation and formatting
- **`StockDataDownloader`** - Main orchestration class

## 📝 Logging

Comprehensive logging to both console and file:

```
2024-01-15 10:30:15 | INFO     | Enhanced Multi-Source Stock Price Data Downloader Started
2024-01-15 10:30:16 | INFO     | ✓ Saved AAPL: 1,234 data points to AAPL.csv
2024-01-15 10:30:45 | INFO     | 🎉 All downloads completed successfully!
```

## 🚨 Error Handling

- **Network errors**: Automatic retry with exponential backoff
- **Invalid tickers**: Validation and filtering with detailed logging
- **API rate limits**: Conservative delays and intelligent chunking
- **Data validation**: Quality checks for downloaded data
- **Resume capability**: Skips existing files automatically

## 📋 Requirements

- Python 3.8+
- pandas
- yfinance
- tqdm

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper type hints and tests
4. Run linting: `black . && isort . && mypy src/`
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Import errors**: Ensure virtual environment is activated and dependencies installed
2. **File not found**: Check that CSV files exist in `data/` directory
3. **Rate limiting**: Increase sleep times if experiencing API errors
4. **Permission errors**: Ensure write permissions for output directories

### Getting Help

1. Check the log file: `multi_source_download.log`
2. Run test version first: `python tests/test_downloader.py`
3. Verify configuration in `src/config.py`
4. Check data source files in `data/` directory

---

**Built with ❤️ for reliable financial data collection** 