#!/usr/bin/env python3
"""
Large Dataset Optimization Script

Prepares the system for efficiently processing large symbol lists like unified_symbols.csv.
Provides configuration recommendations and system checks for optimal performance.
"""

import os
import sys
import psutil
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json

def check_system_resources() -> Dict[str, any]:
    """Check system resources and provide recommendations."""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('.')
    cpu_count = psutil.cpu_count()
    
    return {
        "memory_total_gb": round(memory.total / (1024**3), 1),
        "memory_available_gb": round(memory.available / (1024**3), 1),
        "memory_percent_used": memory.percent,
        "disk_free_gb": round(disk.free / (1024**3), 1),
        "cpu_cores": cpu_count,
        "cpu_logical": psutil.cpu_count(logical=True)
    }

def analyze_symbol_file(file_path: str) -> Dict[str, any]:
    """Analyze the symbol file to understand its characteristics."""
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    try:
        # Read sample to understand structure
        df_sample = pd.read_csv(file_path, nrows=100)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Count total rows (efficient method)
        with open(file_path, 'r') as f:
            total_rows = sum(1 for line in f) - 1  # Subtract header
        
        # Detect ticker column
        ticker_columns = ['ticker', 'symbol', 'Ticker', 'Symbol', 'TICKER', 'SYMBOL']
        ticker_column = None
        for col in ticker_columns:
            if col in df_sample.columns:
                ticker_column = col
                break
        
        # Analyze ticker distribution by exchange/source
        exchange_distribution = {}
        source_distribution = {}
        
        if 'Exchange' in df_sample.columns:
            exchange_dist = df_sample['Exchange'].value_counts()
            exchange_distribution = exchange_dist.to_dict()
        
        if 'Source' in df_sample.columns:
            source_dist = df_sample['Source'].value_counts()
            source_distribution = source_dist.to_dict()
        
        return {
            "file_size_mb": round(file_size / (1024**2), 1),
            "total_symbols": total_rows,
            "columns": list(df_sample.columns),
            "ticker_column": ticker_column,
            "exchange_distribution": exchange_distribution,
            "source_distribution": source_distribution,
            "sample_tickers": df_sample[ticker_column].head(10).tolist() if ticker_column else []
        }
        
    except Exception as e:
        return {"error": f"Error analyzing file: {str(e)}"}

def estimate_processing_time(symbol_count: int, chunk_size: int, sleep_time: int) -> Dict[str, any]:
    """Estimate processing time based on configuration."""
    chunks_needed = (symbol_count + chunk_size - 1) // chunk_size
    
    # Estimate time per ticker (including API calls, processing, saving)
    time_per_ticker = 2.5  # seconds (conservative estimate)
    time_per_chunk = chunk_size * time_per_ticker + sleep_time
    
    total_time_seconds = chunks_needed * time_per_chunk
    total_hours = total_time_seconds / 3600
    
    return {
        "total_symbols": symbol_count,
        "chunks_needed": chunks_needed,
        "estimated_hours": round(total_hours, 1),
        "estimated_days": round(total_hours / 24, 1),
        "time_per_chunk_minutes": round(time_per_chunk / 60, 1)
    }

def generate_optimized_config(symbol_count: int, system_resources: Dict) -> Dict[str, any]:
    """Generate optimized configuration based on symbol count and system resources."""
    
    # Base configuration
    config = {
        "start_date": "2000-01-01",
        "end_date": None,
        "processing_mode": "specific",
        "specific_sources": ["unified_symbols"],
        "log_file": "unified_symbols_download.log"
    }
    
    # Memory-based optimizations
    memory_gb = system_resources["memory_available_gb"]
    
    if memory_gb >= 16:
        # High memory system
        config.update({
            "chunk_size": 15,
            "sleep_between_chunks": 2,
            "individual_sleep": 1,
            "memory_threshold": 0.80,
            "max_parallel_workers": 4
        })
    elif memory_gb >= 8:
        # Medium memory system
        config.update({
            "chunk_size": 10,
            "sleep_between_chunks": 3,
            "individual_sleep": 1,
            "memory_threshold": 0.75,
            "max_parallel_workers": 3
        })
    else:
        # Low memory system
        config.update({
            "chunk_size": 5,
            "sleep_between_chunks": 5,
            "individual_sleep": 2,
            "memory_threshold": 0.70,
            "max_parallel_workers": 2
        })
    
    # Large dataset specific optimizations
    if symbol_count > 20000:
        config.update({
            "chunk_size": max(3, config["chunk_size"] // 2),
            "sleep_between_chunks": config["sleep_between_chunks"] + 1,
            "batch_size": 500,
            "force_gc_interval": 50,
            "early_fallback_threshold": 2
        })
    
    # Add data source configuration
    config["data_sources"] = {
        "unified_symbols": {
            "file": "data/symbol_lists/unified_symbols.csv",
            "ticker_column": "Symbol",
            "description": "Unified Symbol List (Large Dataset)",
            "output_dir": "output/unified_symbols_data",
            "enabled": True
        }
    }
    
    return config

def create_batch_processing_plan(symbol_count: int, batch_size: int = 5000) -> List[Dict]:
    """Create a plan for processing symbols in batches."""
    batches = []
    
    for i in range(0, symbol_count, batch_size):
        start_idx = i
        end_idx = min(i + batch_size, symbol_count)
        batch_num = (i // batch_size) + 1
        
        batches.append({
            "batch_number": batch_num,
            "start_index": start_idx,
            "end_index": end_idx,
            "symbol_count": end_idx - start_idx,
            "output_dir": f"output/unified_symbols_batch_{batch_num:02d}"
        })
    
    return batches

def main():
    """Main optimization analysis and recommendation function."""
    print("🔧 Large Dataset Optimization Analysis")
    print("=" * 60)
    
    # Check system resources
    print("📊 System Resources:")
    resources = check_system_resources()
    print(f"  Memory: {resources['memory_available_gb']:.1f}GB available / {resources['memory_total_gb']:.1f}GB total")
    print(f"  CPU: {resources['cpu_cores']} cores ({resources['cpu_logical']} logical)")
    print(f"  Disk: {resources['disk_free_gb']:.1f}GB free")
    print()
    
    # Analyze symbol file
    symbol_file = "data/symbol_lists/unified_symbols.csv"
    print(f"📈 Analyzing Symbol File: {symbol_file}")
    analysis = analyze_symbol_file(symbol_file)
    
    if "error" in analysis:
        print(f"❌ {analysis['error']}")
        return
    
    print(f"  File size: {analysis['file_size_mb']}MB")
    print(f"  Total symbols: {analysis['total_symbols']:,}")
    print(f"  Ticker column: {analysis['ticker_column']}")
    print(f"  Columns: {', '.join(analysis['columns'])}")
    
    if analysis['exchange_distribution']:
        print(f"  Top exchanges: {dict(list(analysis['exchange_distribution'].items())[:5])}")
    
    print()
    
    # Generate optimized configuration
    print("⚙️  Optimized Configuration:")
    config = generate_optimized_config(analysis['total_symbols'], resources)
    
    print(f"  Chunk size: {config['chunk_size']}")
    print(f"  Sleep between chunks: {config['sleep_between_chunks']}s")
    print(f"  Memory threshold: {config.get('memory_threshold', 0.85)*100:.0f}%")
    print(f"  Parallel workers: {config.get('max_parallel_workers', 3)}")
    print()
    
    # Estimate processing time
    print("⏱️  Processing Time Estimates:")
    estimates = estimate_processing_time(
        analysis['total_symbols'], 
        config['chunk_size'], 
        config['sleep_between_chunks']
    )
    
    print(f"  Total chunks needed: {estimates['chunks_needed']:,}")
    print(f"  Estimated time: {estimates['estimated_hours']:.1f} hours ({estimates['estimated_days']:.1f} days)")
    print(f"  Time per chunk: ~{estimates['time_per_chunk_minutes']:.1f} minutes")
    print()
    
    # Batch processing recommendation
    if analysis['total_symbols'] > 10000:
        print("📦 Batch Processing Recommendation:")
        batches = create_batch_processing_plan(analysis['total_symbols'])
        print(f"  Recommended batches: {len(batches)}")
        print(f"  Symbols per batch: ~{batches[0]['symbol_count']:,}")
        print(f"  Example batches:")
        for batch in batches[:3]:
            print(f"    Batch {batch['batch_number']}: symbols {batch['start_index']:,}-{batch['end_index']:,}")
        if len(batches) > 3:
            print(f"    ... and {len(batches)-3} more batches")
        print()
    
    # Save optimized configuration
    config_file = "config_unified_symbols.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"💾 Optimized configuration saved to: {config_file}")
    print()
    
    # Recommendations
    print("💡 Recommendations:")
    print("  1. Use the generated configuration for optimal performance")
    print("  2. Monitor memory usage during processing")
    print("  3. Consider running during off-peak hours for better API performance")
    print("  4. Ensure stable internet connection for large dataset processing")
    
    if analysis['total_symbols'] > 15000:
        print("  5. Consider batch processing for very large datasets")
        print("  6. Set up resume capability in case of interruptions")
    
    if resources['memory_available_gb'] < 8:
        print("  ⚠️  Warning: Low available memory. Consider closing other applications")
    
    print()
    print("🚀 Ready to process large dataset with optimized settings!")

if __name__ == "__main__":
    main() 