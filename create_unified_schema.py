#!/usr/bin/env python3
"""
Unified Schema Transformation Script

Transforms both stock and crypto data to a unified schema for database integration.
Ensures all data has consistent columns with NULL values where appropriate.

Unified Schema:
- Date, Open, High, Low, Close, Volume (core OHLCV data)
- Symbol, Data_Source, Download_Date (common metadata)
- Adj_Close, Source, Transform_Date, Data_Start_Date, Data_End_Date, Record_Count, Data_Quality_Score, YFinance_Version (stock-specific, NULL for crypto)
- Exchange, Trading_Pair (crypto-specific, NULL for stock)
- Asset_Type (stock/crypto identifier)
"""

import pandas as pd
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import re
from tqdm import tqdm

def setup_logging(log_file: str = "unified_schema_transformation.log") -> logging.Logger:
    """Setup comprehensive logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def detect_data_type(df: pd.DataFrame, filename: str) -> str:
    """
    Detect whether the data is stock or crypto based on columns and filename.
    
    Returns:
        'stock' or 'crypto'
    """
    columns = set(df.columns)
    
    # Check for crypto-specific indicators
    crypto_indicators = {'Exchange', 'Trading_Pair', 'exchange', 'trading_pair'}
    if crypto_indicators.intersection(columns):
        return 'crypto'
    
    # Check for stock-specific indicators
    stock_indicators = {'Adj Close', 'adj_close', 'symbol', 'source', 'yfinance_version'}
    if stock_indicators.intersection(columns):
        return 'stock'
    
    # Check filename patterns
    if any(exchange in filename.lower() for exchange in ['gemini', 'binance', 'bitstamp', 'bitfinex', 'coinbase']):
        return 'crypto'
    
    # Default to stock if unclear
    return 'stock'

def create_unified_schema(df: pd.DataFrame, data_type: str, filename: str) -> pd.DataFrame:
    """
    Transform data to unified schema based on data type.
    
    Unified Schema Columns (in order):
    1. Date (YYYY-MM-DD)
    2. Open (float)
    3. High (float) 
    4. Low (float)
    5. Close (float)
    6. Volume (float)
    7. Symbol (string)
    8. Asset_Type (string: 'stock' or 'crypto')
    9. Data_Source (string)
    10. Download_Date (YYYY-MM-DD)
    11. Adj_Close (float, NULL for crypto)
    12. Source (string, NULL for crypto) 
    13. Transform_Date (YYYY-MM-DD, NULL for crypto)
    14. Data_Start_Date (YYYY-MM-DD, NULL for crypto)
    15. Data_End_Date (YYYY-MM-DD, NULL for crypto)
    16. Record_Count (int, NULL for crypto)
    17. Data_Quality_Score (float, NULL for crypto)
    18. YFinance_Version (string, NULL for crypto)
    19. Exchange (string, NULL for stock)
    20. Trading_Pair (string, NULL for stock)
    """
    
    # Create a copy to avoid modifying original
    unified_df = df.copy()
    
    # Define the unified column order
    unified_columns = [
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol', 'Asset_Type',
        'Data_Source', 'Download_Date', 'Adj_Close', 'Source', 'Transform_Date',
        'Data_Start_Date', 'Data_End_Date', 'Record_Count', 'Data_Quality_Score',
        'YFinance_Version', 'Exchange', 'Trading_Pair'
    ]
    
    if data_type == 'stock':
        # Map stock data columns
        column_mapping = {
            'symbol': 'Symbol',
            'source': 'Source',
            'download_date': 'Download_Date',
            'transform_date': 'Transform_Date',
            'data_start_date': 'Data_Start_Date',
            'data_end_date': 'Data_End_Date',
            'record_count': 'Record_Count',
            'data_quality_score': 'Data_Quality_Score',
            'yfinance_version': 'YFinance_Version',
            'Adj Close': 'Adj_Close'
        }
        
        # Apply column mapping
        for old_col, new_col in column_mapping.items():
            if old_col in unified_df.columns:
                unified_df = unified_df.rename(columns={old_col: new_col})
        
        # Add missing columns with NULL values
        unified_df['Asset_Type'] = 'stock'
        unified_df['Exchange'] = None
        unified_df['Trading_Pair'] = None
        
        # Ensure Data_Source exists
        if 'Data_Source' not in unified_df.columns:
            unified_df['Data_Source'] = unified_df.get('Source', 'Unknown')
    
    elif data_type == 'crypto':
        # Map crypto data columns
        column_mapping = {
            'Trading_Pair': 'Trading_Pair',
            'Data_Source': 'Data_Source',
            'Download_Date': 'Download_Date'
        }
        
        # Apply column mapping
        for old_col, new_col in column_mapping.items():
            if old_col in unified_df.columns:
                unified_df = unified_df.rename(columns={old_col: new_col})
        
        # Add missing columns with NULL values
        unified_df['Asset_Type'] = 'crypto'
        unified_df['Adj_Close'] = None
        unified_df['Source'] = None
        unified_df['Transform_Date'] = None
        unified_df['Data_Start_Date'] = None
        unified_df['Data_End_Date'] = None
        unified_df['Record_Count'] = None
        unified_df['Data_Quality_Score'] = None
        unified_df['YFinance_Version'] = None
    
    # Ensure all required columns exist
    for col in unified_columns:
        if col not in unified_df.columns:
            unified_df[col] = None
    
    # Reorder columns to match unified schema
    unified_df = unified_df[unified_columns]
    
    # Ensure Date column is properly formatted
    if 'Date' in unified_df.columns:
        unified_df['Date'] = pd.to_datetime(unified_df['Date']).dt.strftime('%Y-%m-%d')
    
    # Sort by date (newest first)
    if 'Date' in unified_df.columns:
        unified_df = unified_df.sort_values('Date', ascending=False)
    
    return unified_df

def transform_directory(input_dir: str, output_dir: str, data_type: Optional[str] = None) -> Dict[str, int]:
    """
    Transform all CSV files in input directory to unified schema.
    
    Args:
        input_dir: Directory containing CSV files to transform
        output_dir: Directory to save unified schema files
        data_type: Force data type ('stock' or 'crypto'), or None for auto-detection
    
    Returns:
        Dictionary with transformation statistics
    """
    logger = logging.getLogger(__name__)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files
    csv_files = list(input_path.glob("*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {input_dir}")
        return {"total_files": 0, "successful": 0, "failed": 0, "stock_files": 0, "crypto_files": 0}
    
    logger.info(f"Found {len(csv_files)} CSV files to transform")
    
    stats = {
        "total_files": len(csv_files), 
        "successful": 0, 
        "failed": 0,
        "stock_files": 0,
        "crypto_files": 0
    }
    failed_files = []
    
    # Process files with progress bar
    for csv_file in tqdm(csv_files, desc="Transforming to unified schema"):
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            if df.empty:
                logger.warning(f"Empty file: {csv_file.name} - skipping")
                stats["failed"] += 1
                failed_files.append(csv_file.name)
                continue
            
            # Detect or use specified data type
            detected_type = data_type or detect_data_type(df, csv_file.name)
            
            # Transform to unified schema
            unified_df = create_unified_schema(df, detected_type, csv_file.name)
            
            # Generate output filename
            output_filename = f"unified_{csv_file.name}"
            output_file_path = output_path / output_filename
            
            # Save unified data
            unified_df.to_csv(output_file_path, index=False)
            
            logger.info(f"✓ Transformed {csv_file.name} -> {output_filename} ({detected_type}, {len(unified_df)} rows)")
            stats["successful"] += 1
            stats[f"{detected_type}_files"] += 1
            
        except Exception as e:
            logger.error(f"✗ Failed to transform {csv_file.name}: {str(e)}")
            stats["failed"] += 1
            failed_files.append(csv_file.name)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"UNIFIED SCHEMA TRANSFORMATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total files processed: {stats['total_files']}")
    logger.info(f"Successful transformations: {stats['successful']}")
    logger.info(f"Failed transformations: {stats['failed']}")
    logger.info(f"Stock files: {stats['stock_files']}")
    logger.info(f"Crypto files: {stats['crypto_files']}")
    logger.info(f"Success rate: {(stats['successful']/stats['total_files']*100):.1f}%")
    logger.info(f"Output directory: {output_dir}")
    
    if failed_files:
        logger.warning(f"Failed files: {', '.join(failed_files)}")
    
    return stats

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Transform stock and crypto data to unified schema for database integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform all files with auto-detection
  python create_unified_schema.py --input-dir output/20250601_stock_data --output-dir output/unified_data
  
  # Transform crypto files specifically
  python create_unified_schema.py --input-dir output/transformed_crypto_data --output-dir output/unified_data --data-type crypto
  
  # Transform multiple directories
  python create_unified_schema.py --input-dir output/20250601_stock_data --output-dir output/unified_data
  python create_unified_schema.py --input-dir output/transformed_crypto_data --output-dir output/unified_data --append
        """
    )
    
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing CSV files to transform"
    )
    
    parser.add_argument(
        "--output-dir", 
        required=True,
        help="Directory to save unified schema files"
    )
    
    parser.add_argument(
        "--data-type",
        choices=['stock', 'crypto'],
        help="Force data type (stock or crypto), otherwise auto-detect"
    )
    
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output directory (don't clear it)"
    )
    
    parser.add_argument(
        "--log-file",
        default="unified_schema_transformation.log",
        help="Log file path (default: unified_schema_transformation.log)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    try:
        logger.info(f"Starting unified schema transformation...")
        logger.info(f"Input directory: {args.input_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Data type: {args.data_type or 'auto-detect'}")
        logger.info(f"Append mode: {args.append}")
        
        # Clear output directory if not appending
        if not args.append:
            output_path = Path(args.output_dir)
            if output_path.exists():
                for file in output_path.glob("*.csv"):
                    file.unlink()
                logger.info("Cleared existing output directory")
        
        # Transform data
        stats = transform_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            data_type=args.data_type
        )
        
        logger.info("Unified schema transformation completed successfully!")
        
    except Exception as e:
        logger.error(f"Transformation failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 