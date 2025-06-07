import pandas as pd
import os
import glob

def unify_data_sources():
    """
    Reads all stock/ETF symbol files (.csv, .txt) from the 'data' directory,
    unifies them into a single DataFrame, merges metadata for duplicate
    symbols based on a priority list, and saves the result as a new CSV file.
    """
    data_dir = 'data/symbol_lists/raw_symbol_lists'
    output_filename = 'unified_symbols.csv'
    all_dfs = []

    # --- Configuration ---
    # Define source priority for metadata merging. Richest/most trusted sources first.
    source_priority = ['NYSE', 'NASDAQ', 'AMEX', 'LSE', 'TSX', 'ETF_list', 'IWB_holdings']
    
    # Define mappings for CSV files that have non-standard column names.
    # Format: 'filename.csv': {'symbol_col': 'actual_symbol_col', 'desc_col': 'actual_desc_col'}
    csv_column_mappings = {
        'IWB_holdings_250529.csv': {'symbol_col': 'Ticker', 'desc_col': 'Name'},
        'etf_list.csv': {'symbol_col': 'Symbol', 'desc_col': 'ETF Name'}
    }

    # --- Dynamic File Processing ---
    # Find all .csv and .txt files in the data directory
    files_to_process = glob.glob(os.path.join(data_dir, '*.csv')) + \
                       glob.glob(os.path.join(data_dir, '*.txt'))

    for file_path in files_to_process:
        filename = os.path.basename(file_path)
        source_name = os.path.splitext(filename)[0]

        # Skip the output file to prevent it from reading itself on subsequent runs
        if filename == output_filename:
            continue

        try:
            if filename.endswith('.csv'):
                mapping = csv_column_mappings.get(filename, {})
                symbol_col = mapping.get('symbol_col', 'Symbol')
                desc_col = mapping.get('desc_col', 'Description')

                df = pd.read_csv(file_path)
                df.rename(columns={symbol_col: 'Symbol', desc_col: 'Description'}, inplace=True)
                
            elif filename.endswith('.txt'):
                df = pd.read_csv(file_path, sep='\\t', engine='python')
                # .txt files are assumed to have 'Symbol' and 'Description' columns
            
            else:
                continue

            df['Source'] = source_name if source_name != 'IWB_holdings_250529' else 'IWB_holdings'
            all_dfs.append(df)
            print(f"✓ Processed {filename}")

        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")

    if not all_dfs:
        print("No data files found or processed. Exiting.")
        return

    # --- Unification and Merging ---
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    if 'Symbol' not in combined_df.columns:
        print("Error: 'Symbol' column not found after processing files.")
        return
        
    combined_df['Symbol'] = combined_df['Symbol'].str.strip()
    combined_df.dropna(subset=['Symbol'], inplace=True)
    combined_df = combined_df[combined_df['Symbol'] != '']

    combined_df['source_cat'] = pd.Categorical(combined_df['Source'], categories=source_priority, ordered=True)
    combined_df.sort_values(['Symbol', 'source_cat'], inplace=True)

    agg_funcs = {col: 'first' for col in combined_df.columns if col not in ['Symbol', 'Source', 'source_cat']}
    agg_funcs['Source'] = lambda x: ', '.join(x.unique())

    unified_df = combined_df.groupby('Symbol', as_index=False).agg(agg_funcs)

    # --- Finalizing Output ---
    columns_to_keep = ['Symbol', 'Description', 'Exchange']
    final_columns = [col for col in columns_to_keep if col in unified_df.columns]
    unified_df = unified_df[final_columns]

    output_file = os.path.join('data/symbol_lists', output_filename)
    unified_df.to_csv(output_file, index=False)
    
    print(f"\nSuccessfully created unified CSV with {len(unified_df)} unique symbols.")
    print(f"File saved to: {output_file}")

if __name__ == "__main__":
    unify_data_sources() 