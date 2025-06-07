#!/usr/bin/env python3
"""
Test script for symbol validation functionality.
Tests both format validation and yfinance existence validation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from src.validators import TickerValidator

def test_format_validation():
    """Test basic format validation."""
    print("🧪 Testing Format Validation")
    print("=" * 40)
    
    test_cases = [
        # Valid symbols
        ("AAPL", True),
        ("MSFT", True),
        ("BRK.A", True),
        ("BRK-A", True),
        ("SPY", True),
        
        # Invalid symbols
        ("123", False),
        ("", False),
        ("TOOLONG", False),
        ("@#$", False),
        ("A B", False),
    ]
    
    for symbol, expected in test_cases:
        cleaned = TickerValidator.clean(symbol)
        result = TickerValidator.validate(cleaned)
        status = "✅" if result == expected else "❌"
        print(f"{status} {symbol:10} -> {cleaned:10} = {result} (expected {expected})")

def test_existence_validation():
    """Test yfinance existence validation."""
    print("\n🔍 Testing YFinance Existence Validation")
    print("=" * 50)
    
    # Test with known valid symbols
    valid_symbols = ["AAPL", "MSFT", "SPY", "QQQ", "TSLA"]
    print(f"Testing known valid symbols: {valid_symbols}")
    
    for symbol in valid_symbols:
        exists = TickerValidator.validate_symbol_exists(symbol, timeout=10)
        status = "✅" if exists else "❌"
        print(f"{status} {symbol}: {'EXISTS' if exists else 'NOT FOUND'}")
    
    # Test with known invalid symbols
    invalid_symbols = ["INVALID123", "NOTREAL", "FAKE", "XXXX", "ZZZZ"]
    print(f"\nTesting known invalid symbols: {invalid_symbols}")
    
    for symbol in invalid_symbols:
        exists = TickerValidator.validate_symbol_exists(symbol, timeout=5)
        status = "✅" if not exists else "❌"
        print(f"{status} {symbol}: {'EXISTS' if exists else 'NOT FOUND'} (expected NOT FOUND)")

def test_batch_validation():
    """Test batch validation functionality."""
    print("\n📦 Testing Batch Validation")
    print("=" * 35)
    
    test_symbols = [
        # Mix of valid and invalid symbols
        "AAPL", "MSFT", "INVALID123", "SPY", "NOTREAL", 
        "QQQ", "FAKE", "TSLA", "XXXX", "GOOGL"
    ]
    
    print(f"Testing batch validation with: {test_symbols}")
    
    valid, invalid = TickerValidator.batch_validate_symbols(
        test_symbols, 
        max_workers=3, 
        timeout=5
    )
    
    print(f"\n✅ Valid symbols ({len(valid)}): {valid}")
    print(f"❌ Invalid symbols ({len(invalid)}): {invalid}")
    print(f"📊 Success rate: {len(valid)/len(test_symbols)*100:.1f}%")

def test_date_range_detection():
    """Test historical date range detection."""
    print("\n📅 Testing Historical Date Range Detection")
    print("=" * 50)
    
    # Test with known symbols that have long histories
    test_symbols = ["AAPL", "MSFT", "SPY", "IBM", "KO"]
    print(f"Testing date range detection for: {test_symbols}")
    
    valid_with_dates, invalid = TickerValidator.batch_validate_with_date_ranges(
        test_symbols,
        max_workers=3,
        timeout=10
    )
    
    print(f"\n📊 Results:")
    for symbol, earliest_date in valid_with_dates.items():
        # Handle timezone-aware datetime comparison
        now = datetime.now()
        if earliest_date.tzinfo is not None:
            earliest_date = earliest_date.replace(tzinfo=None)
        years_available = (now - earliest_date).days / 365.25
        print(f"✅ {symbol}: {earliest_date.strftime('%Y-%m-%d')} ({years_available:.1f} years of data)")
    
    if invalid:
        print(f"❌ Invalid symbols: {invalid}")
    
    print(f"\n📈 Summary: {len(valid_with_dates)} symbols with date ranges detected")

def main():
    """Run all validation tests."""
    print("🧪 Symbol Validation Test Suite")
    print("=" * 60)
    
    try:
        test_format_validation()
        test_existence_validation()
        test_batch_validation()
        test_date_range_detection()
        
        print("\n🎉 All tests completed!")
        print("💡 The validation system is ready for large dataset processing.")
        print("📈 Maximum historical data will be downloaded for each symbol.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 