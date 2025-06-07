#!/usr/bin/env python3
"""
Stock Market Data Transformation Script

This script transforms historical stock market data by adding required columns:
1. symbol - extracted from the CSV filename
2. source - determined by the output directory structure
3. download_date - when the data was originally downloaded (parameterizable)
4. transform_date - when this transformation was applied
5. data_start_date - first date in the time series
6. data_end_date - last date in the time series
7. record_count - number of data points
8. data_quality_score - percentage of complete OHLCV data
9. yfinance_version - version info for reproducibility

The script handles multiple data sources with different formats and automatically
detects and processes all CSV files in the output directory structure.
"""

import os
import pandas as pd
from datetime import datetime, date
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import argparse

# Try modern importlib.metadata first, fallback to pkg_resources
try:
    from importlib.metadata import version
    def get_package_version(package_name):
        try:
            return version(package_name)
        except:
            return "unknown"
except ImportError:
    try:
        import pkg_resources
        def get_package_version(package_name):
            try:
                return pkg_resources.get_distribution(package_name).version
            except:
                return "unknown"
    except ImportError:
        def get_package_version(package_name):
            return "unknown"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_transformation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataTransformer:
    """Handles transformation of stock market data with source detection and comprehensive metadata addition."""
    
    def __init__(self, output_dir: str = "output", backup_dir: str = "backup", download_date: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.backup_dir = Path(backup_dir)
        self.transform_date = datetime.now().strftime("%Y-%m-%d")
        
        # Handle download date parameter
        if download_date:
            try:
                # Validate date format
                datetime.strptime(download_date, "%Y-%m-%d")
                self.download_date = download_date
            except ValueError:
                raise ValueError(f"Invalid date format: {download_date}. Use YYYY-MM-DD format.")
        else:
            self.download_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get yfinance version for reproducibility
        self.yfinance_version = get_package_version("yfinance")
        
        # Source mapping based on directory names
        self.source_mapping = {
            "etf_list_data": "etf_list",
            "iwb_holdings_250529_data": "iwb_holdings_250529",
            "stock_data": "iwb_holdings",  # Legacy directory
            "etf_data": "etf_list",        # Legacy directory
        }
        
        # Statistics tracking
        self.stats = {
            "total_files": 0,
            "successful_transformations": 0,
            "failed_transformations": 0,
            "sources_processed": set(),
            "files_by_source": {},
            "quality_stats": {
                "high_quality": 0,    # >95% complete data
                "medium_quality": 0,  # 80-95% complete data
                "low_quality": 0      # <80% complete data
            }
        }
    
    def detect_data_sources(self) -> Dict[str, List[Path]]:
        """
        Detect all data source directories and their CSV files.
        
        Returns:
            Dictionary mapping source names to lists of CSV file paths
        """
        sources = {}
        
        if not self.output_dir.exists():
            logger.error(f"Output directory {self.output_dir} does not exist")
            return sources
        
        # Scan for data directories
        for item in self.output_dir.iterdir():
            if item.is_dir():
                csv_files = list(item.glob("*.csv"))
                if csv_files:
                    source_name = self.source_mapping.get(item.name, item.name)
                    sources[source_name] = csv_files
                    logger.info(f"Found {len(csv_files)} CSV files in {item.name} (source: {source_name})")
        
        return sources
    
    def extract_symbol_from_filename(self, file_path: Path) -> str:
        """
        Extract ticker symbol from CSV filename.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Ticker symbol (filename without extension)
        """
        return file_path.stem
    
    def calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate data quality score based on completeness of OHLCV data.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Quality score as percentage (0-100)
        """
        if df.empty:
            return 0.0
        
        # Check completeness of key columns
        key_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in key_columns if col in df.columns]
        
        if not available_columns:
            return 0.0
        
        # Calculate percentage of non-null values across key columns
        total_cells = len(df) * len(available_columns)
        non_null_cells = df[available_columns].count().sum()
        
        return (non_null_cells / total_cells) * 100 if total_cells > 0 else 0.0
    
    def get_date_range(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the start and end dates from the time series data.
        
        Args:
            df: DataFrame with Date column
            
        Returns:
            Tuple of (start_date, end_date) as strings, or (None, None) if no valid dates
        """
        if 'Date' not in df.columns or df.empty:
            return None, None
        
        try:
            # Convert Date column to datetime
            dates = pd.to_datetime(df['Date'])
            start_date = dates.min().strftime("%Y-%m-%d")
            end_date = dates.max().strftime("%Y-%m-%d")
            return start_date, end_date
        except:
            return None, None
    
    def validate_csv_structure(self, df: pd.DataFrame, file_path: Path) -> bool:
        """
        Validate that the CSV has the expected OHLCV structure.
        
        Args:
            df: DataFrame to validate
            file_path: Path to the file being validated
            
        Returns:
            True if structure is valid, False otherwise
        """
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"File {file_path.name} missing columns: {missing_columns}")
            return False
        
        if df.empty:
            logger.warning(f"File {file_path.name} is empty")
            return False
        
        return True
    
    def transform_csv_file(self, file_path: Path, source: str) -> bool:
        """
        Transform a single CSV file by adding comprehensive metadata columns.
        
        Args:
            file_path: Path to the CSV file
            source: Source identifier for the data
            
        Returns:
            True if transformation successful, False otherwise
        """
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Validate structure
            if not self.validate_csv_structure(df, file_path):
                return False
            
            # Extract symbol from filename
            symbol = self.extract_symbol_from_filename(file_path)
            
            # Calculate metadata
            data_quality_score = self.calculate_data_quality_score(df)
            start_date, end_date = self.get_date_range(df)
            record_count = len(df)
            
            # Add comprehensive metadata columns
            df['symbol'] = symbol
            df['source'] = source
            df['download_date'] = self.download_date
            df['transform_date'] = self.transform_date
            df['data_start_date'] = start_date
            df['data_end_date'] = end_date
            df['record_count'] = record_count
            df['data_quality_score'] = round(data_quality_score, 2)
            df['yfinance_version'] = self.yfinance_version
            
            # Reorder columns to put metadata at the beginning
            metadata_columns = [
                'symbol', 'source', 'download_date', 'transform_date',
                'data_start_date', 'data_end_date', 'record_count', 
                'data_quality_score', 'yfinance_version'
            ]
            original_columns = [col for col in df.columns if col not in metadata_columns]
            new_column_order = metadata_columns + original_columns
            df = df[new_column_order]
            
            # Update quality statistics
            if data_quality_score >= 95:
                self.stats["quality_stats"]["high_quality"] += 1
            elif data_quality_score >= 80:
                self.stats["quality_stats"]["medium_quality"] += 1
            else:
                self.stats["quality_stats"]["low_quality"] += 1
            
            # Create backup if requested
            if self.backup_dir:
                self.create_backup(file_path)
            
            # Save the transformed file
            df.to_csv(file_path, index=False)
            
            logger.info(f"✓ Transformed {file_path.name}: {record_count} rows, quality={data_quality_score:.1f}%, symbol={symbol}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to transform {file_path.name}: {str(e)}")
            return False
    
    def create_backup(self, file_path: Path) -> None:
        """
        Create a backup of the original file before transformation.
        
        Args:
            file_path: Path to the file to backup
        """
        try:
            # Create backup directory structure
            relative_path = file_path.relative_to(self.output_dir)
            backup_path = self.backup_dir / relative_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file to backup location
            import shutil
            shutil.copy2(file_path, backup_path)
            
        except Exception as e:
            logger.warning(f"Failed to create backup for {file_path.name}: {str(e)}")
    
    def transform_all_data(self, create_backup: bool = True) -> None:
        """
        Transform all detected data sources.
        
        Args:
            create_backup: Whether to create backups before transformation
        """
        logger.info("Starting comprehensive data transformation process...")
        logger.info(f"Download date: {self.download_date}")
        logger.info(f"Transform date: {self.transform_date}")
        logger.info(f"YFinance version: {self.yfinance_version}")
        
        if create_backup:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Backups will be created in: {self.backup_dir}")
        else:
            self.backup_dir = None
        
        # Detect all data sources
        sources = self.detect_data_sources()
        
        if not sources:
            logger.error("No data sources found!")
            return
        
        # Process each source
        for source_name, csv_files in sources.items():
            logger.info(f"\nProcessing source: {source_name}")
            logger.info(f"Files to process: {len(csv_files)}")
            
            self.stats["sources_processed"].add(source_name)
            self.stats["files_by_source"][source_name] = {"total": len(csv_files), "successful": 0, "failed": 0}
            
            for csv_file in csv_files:
                self.stats["total_files"] += 1
                
                if self.transform_csv_file(csv_file, source_name):
                    self.stats["successful_transformations"] += 1
                    self.stats["files_by_source"][source_name]["successful"] += 1
                else:
                    self.stats["failed_transformations"] += 1
                    self.stats["files_by_source"][source_name]["failed"] += 1
        
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print comprehensive transformation summary statistics."""
        logger.info("\n" + "="*70)
        logger.info("COMPREHENSIVE TRANSFORMATION SUMMARY")
        logger.info("="*70)
        logger.info(f"Total files processed: {self.stats['total_files']}")
        logger.info(f"Successful transformations: {self.stats['successful_transformations']}")
        logger.info(f"Failed transformations: {self.stats['failed_transformations']}")
        logger.info(f"Success rate: {(self.stats['successful_transformations']/self.stats['total_files']*100):.1f}%")
        logger.info(f"Sources processed: {len(self.stats['sources_processed'])}")
        logger.info(f"Download date: {self.download_date}")
        logger.info(f"Transform date: {self.transform_date}")
        logger.info(f"YFinance version: {self.yfinance_version}")
        
        logger.info("\nData Quality Distribution:")
        total_quality_files = sum(self.stats["quality_stats"].values())
        if total_quality_files > 0:
            for quality, count in self.stats["quality_stats"].items():
                percentage = (count / total_quality_files) * 100
                logger.info(f"  {quality.replace('_', ' ').title()}: {count} files ({percentage:.1f}%)")
        
        logger.info("\nBreakdown by source:")
        for source, counts in self.stats["files_by_source"].items():
            success_rate = (counts["successful"] / counts["total"] * 100) if counts["total"] > 0 else 0
            logger.info(f"  {source}: {counts['successful']}/{counts['total']} files ({success_rate:.1f}%)")
    
    def sample_transformed_data(self, num_samples: int = 3) -> None:
        """
        Display sample of transformed data for verification.
        
        Args:
            num_samples: Number of sample files to display
        """
        logger.info(f"\nSample of transformed data (showing {num_samples} files):")
        
        sources = self.detect_data_sources()
        sample_count = 0
        
        for source_name, csv_files in sources.items():
            if sample_count >= num_samples:
                break
                
            for csv_file in csv_files[:min(2, len(csv_files))]:
                if sample_count >= num_samples:
                    break
                    
                try:
                    df = pd.read_csv(csv_file)
                    logger.info(f"\nFile: {csv_file.name}")
                    logger.info(f"Columns: {list(df.columns)}")
                    logger.info(f"Shape: {df.shape}")
                    logger.info("First few rows:")
                    logger.info(df.head(3).to_string())
                    sample_count += 1
                    
                except Exception as e:
                    logger.error(f"Error reading sample file {csv_file.name}: {str(e)}")

def main():
    """Main function with enhanced command-line interface."""
    parser = argparse.ArgumentParser(
        description="Transform stock market data by adding comprehensive metadata columns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform data downloaded yesterday (2025-06-01)
  python transform_data.py --download-date 2025-06-01
  
  # Dry run to see what would be processed
  python transform_data.py --dry-run
  
  # Transform with custom directories and show samples
  python transform_data.py --output-dir my_data --sample
  
  # Transform without backups
  python transform_data.py --no-backup --download-date 2025-06-01
        """
    )
    parser.add_argument(
        "--output-dir", 
        default="output", 
        help="Directory containing the data to transform (default: output)"
    )
    parser.add_argument(
        "--backup-dir", 
        default="backup", 
        help="Directory to store backups (default: backup)"
    )
    parser.add_argument(
        "--download-date",
        help="Date when the data was originally downloaded (YYYY-MM-DD format). Defaults to current date."
    )
    parser.add_argument(
        "--no-backup", 
        action="store_true", 
        help="Skip creating backups"
    )
    parser.add_argument(
        "--sample", 
        action="store_true", 
        help="Show sample of transformed data after processing"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be transformed without making changes"
    )
    
    args = parser.parse_args()
    
    # Initialize transformer
    try:
        transformer = DataTransformer(
            output_dir=args.output_dir,
            backup_dir=args.backup_dir if not args.no_backup else None,
            download_date=args.download_date
        )
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be modified")
        sources = transformer.detect_data_sources()
        
        logger.info(f"\nWould transform data in: {args.output_dir}")
        logger.info(f"Download date would be set to: {transformer.download_date}")
        logger.info(f"Transform date would be: {transformer.transform_date}")
        logger.info(f"YFinance version: {transformer.yfinance_version}")
        
        for source_name, csv_files in sources.items():
            logger.info(f"\n  Source '{source_name}': {len(csv_files)} files")
            for csv_file in csv_files[:5]:  # Show first 5 files
                symbol = transformer.extract_symbol_from_filename(csv_file)
                logger.info(f"    - {csv_file.name} → symbol={symbol}, source={source_name}")
            if len(csv_files) > 5:
                logger.info(f"    ... and {len(csv_files) - 5} more files")
    else:
        # Perform transformation
        transformer.transform_all_data(create_backup=not args.no_backup)
        
        # Show sample if requested
        if args.sample:
            transformer.sample_transformed_data()
    
    return 0

if __name__ == "__main__":
    exit(main()) 