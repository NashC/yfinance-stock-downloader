# Enhanced Multi-Source Stock Price Data Downloader

A production-ready Python application for downloading historical stock price data from Yahoo Finance with **complete generalization** for any CSV ticker sources. No longer limited to specific files - works with any CSV containing ticker symbols!

## 🚀 Key Features

- **🔍 Auto-Discovery** - Automatically detects CSV files and ticker columns
- **📄 JSON Configuration** - Flexible configuration for complex setups  
- **🔄 Legacy Compatibility** - Backward compatible with original hardcoded sources
- **🎯 Selective Processing** - Process specific sources or all available
- **📊 Real-time Progress** - Progress bars and comprehensive logging
- **🔁 Resume Capability** - Automatically skips already downloaded files
- **✅ Data Validation** - Validates ticker symbols and data quality
- **🏗️ Modular Architecture** - Clean separation with organized packages
- **⚡ Intelligent Rate Limiting** - Respects Yahoo Finance API limits
- **🧪 Comprehensive Testing** - Full test suite with multiple scenarios

## 📁 Project Structure

```
yfinance_stock_downloader/
├── src/                          # Main package
│   ├── __init__.py              # Package exports
│   ├── config.py                # Generalized configuration system
│   ├── validators.py            # Ticker and data validation
│   └── downloader.py            # Main downloader class
├── tests/                       # Test suite
│   ├── test_downloader.py       # Comprehensive test suite
│   ├── test_outputs/            # Test data outputs
│   └── logs/                    # Test logs
├── data/                        # Input CSV files (auto-discovered)
│   ├── IWB_holdings_250529.csv # Example: Russell 1000 stocks
│   └── etf_list.csv            # Example: ETF universe
├── main.py                      # Main entry point with CLI
├── config.json                  # Sample JSON configuration
├── pyproject.toml              # Modern Python packaging
└── README.md                   # This file
```

## 🎯 Multiple Usage Modes

### 1. **Auto-Discovery Mode (Recommended)**
```bash
# Automatically discover CSV files in data/ directory
.venv/bin/python main.py

# Use custom directories
.venv/bin/python main.py --data-dir my_data --output-dir my_output
```

**What it does:**
- Scans for CSV files in the specified directory
- Auto-detects ticker columns (`ticker`, `symbol`, `Ticker`, `Symbol`, etc.)
- Creates organized output directories for each source
- Processes all discovered sources

### 2. **JSON Configuration Mode**
```bash
# Create sample configuration
.venv/bin/python main.py --create-config

# Use custom configuration
.venv/bin/python main.py --config config.json
```

**Sample JSON Configuration:**
```json
{
  "start_date": "2000-01-01",
  "end_date": null,
  "processing_mode": "all",
  "chunk_size": 25,
  "sleep_between_chunks": 5,
  "data_sources": {
    "sp500": {
      "file": "data/sp500_tickers.csv",
      "ticker_column": "Symbol",
      "description": "S&P 500 Companies",
      "output_dir": "output/sp500_data",
      "enabled": true,
      "filter_column": "Market Cap",
      "filter_values": ["Large Cap"]
    }
  }
}
```

### 3. **Legacy Mode (Backward Compatible)**
```bash
# Use original hardcoded configuration
.venv/bin/python main.py --legacy
```

### 4. **Selective Processing**
```bash
# Process only specific sources
.venv/bin/python main.py --sources etf_list,iwb_holdings

# Process only ETFs with custom settings
.venv/bin/python main.py --sources etf_list --chunk-size 10 --sleep 3
```

## 🛠️ Installation & Setup

### Modern Setup with uv (Recommended)
```bash
# Clone repository
git clone https://github.com/NashC/yfinance-stock-downloader.git
cd yfinance-stock-downloader

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install project with dependencies
uv pip install -e .
```

### Alternative Installation
```bash
# Install dependencies only
uv pip install -r requirements.txt

# Use lock file for exact reproducibility
uv pip sync requirements.lock

# Install with development tools
uv pip install -e ".[dev]"
```

### Legacy Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📊 Data Sources & Configuration

### **Supported CSV Formats**
The system automatically detects CSV files with ticker symbols in columns named:
- `ticker`, `Ticker`, `TICKER`
- `symbol`, `Symbol`, `SYMBOL`
- Any column specified in JSON configuration

### **Example Data Sources**
- **Individual Stocks**: Russell 1000, S&P 500, NASDAQ 100, custom stock lists
- **ETFs**: Broad market, sector, international, bond, commodity ETFs
- **Custom Lists**: Any CSV with ticker symbols

### **Output Structure**
```
output/
├── source1_data/
│   ├── AAPL.csv
│   ├── MSFT.csv
│   └── ...
├── source2_data/
│   ├── VOO.csv
│   ├── SPY.csv
│   └── ...
└── logs/
    └── stock_download.log
```

## 🎮 Command-Line Interface

```bash
# Show all options
.venv/bin/python main.py --help

# Common usage patterns
.venv/bin/python main.py                           # Auto-discover and process all
.venv/bin/python main.py --config my_config.json   # Use JSON configuration
.venv/bin/python main.py --legacy                  # Original behavior
.venv/bin/python main.py --create-config           # Create sample config
.venv/bin/python main.py --sources etfs            # Process specific sources
.venv/bin/python main.py --start-date 2020-01-01   # Custom date range
.venv/bin/python main.py --chunk-size 10 --sleep 2 # Performance tuning
```

## 📈 Progress Tracking

The system provides comprehensive progress updates:

```
🔍 Auto-discovering CSV files in 'data' directory...
✓ Auto-discovered: etf_list.csv (ticker column: Symbol)
✓ Auto-discovered: IWB_holdings_250529.csv (ticker column: Ticker)

✓ Found 2 data source(s):
  • Auto-discovered: etf_list.csv
    File: data/etf_list.csv
    Ticker column: Symbol
    Output: output/etf_list_data

INFO     | Enhanced Multi-Source Stock Price Data Downloader Started
INFO     | Processing Mode: all
INFO     | Configuration: START_DATE=2000-01-01, END_DATE=None

Downloading ETF Universe: 45%|████▌     | 112/250 [02:15<02:45, 1.2ticker/s]

INFO     | ✓ Saved VOO: 6234 data points to VOO.csv
INFO     | ✓ Chunk completed: 25 successful, 0 failed

COMPREHENSIVE DOWNLOAD SUMMARY
================================================================================
ETF Universe:
  • Total tickers: 250
  • Successfully downloaded: 248
  • Failed: 2
  • Success rate: 99.2%
  • Output directory: output/etf_list_data
```

## 🧪 Testing

### Run Comprehensive Tests
```bash
# Run full test suite
.venv/bin/python tests/test_downloader.py

# Test specific functionality
.venv/bin/python -c "from src.config import ConfigLoader; ConfigLoader.auto_discover_csv_files('data')"
```

### Test Coverage
- ✅ Auto-discovery functionality
- ✅ JSON configuration loading
- ✅ Legacy compatibility
- ✅ Limited download testing
- ✅ Error handling and retry logic
- ✅ Progress tracking and logging

## 🔧 Configuration Options

### **Processing Modes**
- `all` - Process all available sources (default)
- `specific` - Process only named sources
- `auto` - Auto-detect and process available files

### **Performance Tuning**
- `chunk_size` - Number of tickers per batch (default: 25)
- `sleep_between_chunks` - Seconds between chunks (default: 5)
- `individual_sleep` - Seconds between individual downloads (default: 2)
- `max_retries` - Maximum retry attempts (default: 3)

### **Data Validation**
- `min_data_points` - Minimum data points to consider valid (default: 10)
- Automatic ticker symbol validation
- Data quality checks and cleaning

### **Advanced Features**
- **Filtering**: Filter CSV rows based on column values
- **Custom Output**: Specify output directories per source
- **Resume Capability**: Automatically skip existing files
- **Comprehensive Logging**: Detailed logs with timestamps

## 🚨 Error Handling

- **Network errors**: Automatic retry with exponential backoff
- **Invalid tickers**: Validation and filtering with detailed logging
- **API rate limits**: Conservative delays and intelligent chunking
- **Data validation**: Quality checks for downloaded data
- **File conflicts**: Resume capability skips existing files

## 📋 Requirements

- Python 3.9+
- pandas>=2.0.0
- yfinance>=0.2.18
- tqdm>=4.65.0

## 🆕 Migration from Original Version

### **Automatic Migration**
The system is **100% backward compatible**:
```bash
# Original behavior (still works)
.venv/bin/python main.py --legacy
```

### **Recommended Upgrade Path**
1. **Test auto-discovery**: `.venv/bin/python main.py`
2. **Create custom config**: `.venv/bin/python main.py --create-config`
3. **Customize as needed**: Edit `config.json`
4. **Use new features**: Selective processing, custom sources, etc.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper type hints and tests
4. Run tests: `.venv/bin/python tests/test_downloader.py`
5. Run linting: `black . && isort . && mypy src/`
6. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Virtual Environment**: Ensure you're using `.venv/bin/python` or activate the environment
2. **Dependencies**: Install with `uv pip install -e .`
3. **CSV Format**: Ensure CSV has recognizable ticker column names
4. **Permissions**: Ensure write permissions for output directories

### Getting Help

1. **Check logs**: Review the detailed log files
2. **Run tests**: `.venv/bin/python tests/test_downloader.py`
3. **Test auto-discovery**: Verify CSV files are detected correctly
4. **Use legacy mode**: Fall back to original behavior if needed

### Debug Commands
```bash
# Test auto-discovery
.venv/bin/python -c "from src.config import ConfigLoader; ConfigLoader.auto_discover_csv_files('data')"

# Create sample config
.venv/bin/python main.py --create-config

# Test with minimal data
.venv/bin/python main.py --sources etf_list --chunk-size 2
```

---

**🎯 Now supports ANY CSV files with ticker symbols - not just the original two sources!**

Built with ❤️ for flexible, scalable financial data collection 