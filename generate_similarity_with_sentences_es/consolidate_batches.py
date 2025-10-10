import pandas as pd
import glob
import re
import os

# Configuration
SUBSET_LANG = "es"
OUTPUT_CSV = f"similarity_pairs_{SUBSET_LANG}_consolidated.csv"

def extract_batch_number(filename):
    """Extract the starting batch number from filename for sorting"""
    match = re.search(r'batch_(\d+)_\d+\.xlsx$', filename)
    return int(match.group(1)) if match else 0

def consolidate_batch_files():
    """Read all batch files in order and consolidate into a single CSV"""
    
    # Find all batch files
    pattern = f"similarity_pairs_{SUBSET_LANG}_batch_*.xlsx"
    batch_files = glob.glob(pattern)
    
    if not batch_files:
        print(f"No batch files found with pattern: {pattern}")
        return
    
    # Sort files by batch number
    batch_files.sort(key=extract_batch_number)
    
    print(f"Found {len(batch_files)} batch files:")
    for file in batch_files:
        print(f"  - {file}")
    
    consolidated_data = []
    total_pairs = 0
    
    # Read each batch file
    for i, batch_file in enumerate(batch_files):
        try:
            print(f"Reading {batch_file}...")
            df = pd.read_excel(batch_file, sheet_name="similarity_pairs")
            
            batch_pairs = len(df)
            total_pairs += batch_pairs
            print(f"  Loaded {batch_pairs} pairs")
            
            consolidated_data.append(df)
            
        except Exception as e:
            print(f"Error reading {batch_file}: {e}")
            continue
    
    if not consolidated_data:
        print("No data could be loaded from batch files.")
        return
    
    # Concatenate all dataframes
    print("\nConcatenating all batch data...")
    final_df = pd.concat(consolidated_data, ignore_index=True)
    
    print(f"Total pairs before final deduplication: {len(final_df)}")
    
    # Apply final deduplication (in case there are overlaps between batches)
    initial_count = len(final_df)
    
    # Remove self-comparisons
    final_df = final_df[final_df['idx1'] != final_df['idx2']]
    after_self_removal = len(final_df)
    
    # Remove symmetric duplicates
    final_df['pair_sorted'] = final_df[['idx1', 'idx2']].apply(lambda x: tuple(sorted(x)), axis=1)
    final_df = final_df.drop_duplicates(subset=['pair_sorted'], keep='first')
    final_df = final_df.drop('pair_sorted', axis=1)
    
    final_count = len(final_df)
    
    print(f"Removed {initial_count - after_self_removal} self-comparison pairs")
    print(f"Removed {after_self_removal - final_count} symmetric duplicate pairs")
    print(f"Final unique pairs: {final_count}")
    
    # Save to CSV
    print(f"\nSaving consolidated data to {OUTPUT_CSV}...")
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"Successfully saved {len(final_df)} pairs to {OUTPUT_CSV}")
    
    # Show file size
    file_size = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)  # MB
    print(f"Output file size: {file_size:.2f} MB")
    
    # Show some sample data
    print(f"\nSample of consolidated data:")
    print(final_df.head())
    
    return final_df

if __name__ == "__main__":
    print("Consolidating similarity batch files...")
    print("=" * 50)
    
    result = consolidate_batch_files()
    
    if result is not None:
        print("=" * 50)
        print("Consolidation completed successfully!")
    else:
        print("Consolidation failed.")