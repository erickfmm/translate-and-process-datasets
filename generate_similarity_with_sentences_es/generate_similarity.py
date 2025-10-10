from datasets import load_dataset
import pandas as pd
import numpy as np
import random
from typing import List, Tuple

DATASET_SENTENCES = "agentlans/multilingual-sentences"
SPLIT = "train"
SUBSET_LANG = "es"
EMBEDDING_MODEL = "hiiamsid/sentence_similarity_spanish_es" #Spanish sentence similarity model
N_ROWS_PER_BATCH = 1000 #number of rows to process in each batch
N_REPETITIONS_PER_ROW = 1 #number of times each row is compared to other rows
N_ROWS_PER_ROW = 10 #number of other rows to compare each row to


from sentence_transformers import SentenceTransformer, util

def generate_similarity_pairs(sentences: List[str], start_idx: int, model: SentenceTransformer) -> List[Tuple[int, int, str, str, float]]:
    """
    Generate similarity pairs for sentences based on the configured parameters.
    Returns list of tuples: (idx1, idx2, sentence1, sentence2, similarity_score)
    """
    embeddings = model.encode(sentences, convert_to_tensor=True)
    
    pairs = []
    n_sentences = len(sentences)
    
    for i, sentence1 in enumerate(sentences):
        actual_idx1 = start_idx + i
        
        # For each repetition
        for rep in range(N_REPETITIONS_PER_ROW):
            # Get random indices for comparison (excluding self)
            available_indices = [j for j in range(n_sentences) if j != i]
            
            if len(available_indices) >= N_ROWS_PER_ROW:
                comparison_indices = random.sample(available_indices, N_ROWS_PER_ROW)
            else:
                comparison_indices = available_indices
            
            for j in comparison_indices:
                actual_idx2 = start_idx + j
                sentence2 = sentences[j]
                
                # Additional safeguard: skip if comparing same indices
                if actual_idx1 == actual_idx2:
                    continue
                
                # Compute cosine similarity
                similarity = util.cos_sim(embeddings[i:i+1], embeddings[j:j+1]).item()
                
                pairs.append((actual_idx1, actual_idx2, sentence1, sentence2, similarity))
    
    return pairs

try:
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Loaded embedding model: {EMBEDDING_MODEL}")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    exit(1)

try:
    # Login using e.g. `huggingface-cli login` to access this dataset
    print(f"Attempting to load dataset: {DATASET_SENTENCES}, subset: {SUBSET_LANG}, split: {SPLIT}")
    ds = load_dataset(DATASET_SENTENCES, SUBSET_LANG, split=SPLIT)
    print(f"Loaded {len(ds)} rows from {DATASET_SENTENCES}:{SUBSET_LANG}:{SPLIT}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise e

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

all_similarity_pairs = []

for i in range(0, len(ds), N_ROWS_PER_BATCH):
    try:
        batch = ds[i:i+N_ROWS_PER_BATCH]
        sentences = batch["text"]
        print(f"Processing rows {i} to {i+len(sentences)}...")
        
        # Generate similarity pairs for this batch
        batch_pairs = generate_similarity_pairs(sentences, i, model)
        all_similarity_pairs.extend(batch_pairs)
        
        print(f"Generated {len(batch_pairs)} similarity pairs for this batch")
        
        # Save batch results immediately to avoid data loss
        if batch_pairs:
            try:
                batch_df = pd.DataFrame(batch_pairs, columns=['idx1', 'idx2', 'sentence1', 'sentence2', 'similarity'])
                
                # Apply deduplication to batch
                batch_df = batch_df[batch_df['idx1'] != batch_df['idx2']]
                batch_df['pair_sorted'] = batch_df[['idx1', 'idx2']].apply(lambda x: tuple(sorted(x)), axis=1)
                batch_df = batch_df.drop_duplicates(subset=['pair_sorted'], keep='first')
                batch_df = batch_df.drop('pair_sorted', axis=1)
                
                # Save batch file
                batch_output_file = f"similarity_pairs_{SUBSET_LANG}_batch_{i}_{i+len(sentences)}.xlsx"
                batch_df.to_excel(batch_output_file, sheet_name="similarity_pairs", index=False)
                print(f"Saved batch to {batch_output_file} ({len(batch_df)} pairs)")
                
            except Exception as batch_save_error:
                print(f"Warning: Could not save batch {i}: {batch_save_error}")
                # Continue processing even if batch save fails
        
    except Exception as e:
        print(f"Error processing batch {i}: {e}")
        exit(1)

print(f"Total similarity pairs generated: {len(all_similarity_pairs)}")

# Convert to DataFrame and efficiently drop self-comparisons and duplicates
try:
    df = pd.DataFrame(all_similarity_pairs, columns=['idx1', 'idx2', 'sentence1', 'sentence2', 'similarity'])
    
    # Efficiently drop rows where idx1 == idx2 (self-comparisons)
    initial_count = len(df)
    df = df[df['idx1'] != df['idx2']]
    after_self_removal = len(df)
    
    # Remove symmetric duplicates (e.g., (4,8) and (8,4))
    # Create sorted pair columns to identify duplicates regardless of order
    df['pair_sorted'] = df[['idx1', 'idx2']].apply(lambda x: tuple(sorted(x)), axis=1)
    
    # Keep only the first occurrence of each unique pair
    df_deduplicated = df.drop_duplicates(subset=['pair_sorted'], keep='first')
    
    # Drop the helper column
    df_deduplicated = df_deduplicated.drop('pair_sorted', axis=1)
    
    final_count = len(df_deduplicated)
    
    print(f"Dropped {initial_count - after_self_removal} self-comparison pairs")
    print(f"Dropped {after_self_removal - final_count} symmetric duplicate pairs")
    print(f"Remaining unique pairs: {final_count}")
    
    # Save consolidated results
    output_file = f"similarity_pairs_{SUBSET_LANG}_consolidated.xlsx"
    print(f"Saving consolidated {len(df_deduplicated)} similarity pairs to {output_file}...")
    df_deduplicated.to_excel(output_file, sheet_name="similarity_pairs", index=False)
    print("Done.")
    
    # Optional: Create a summary of all batch files created
    print(f"\nBatch files created for safety:")
    import glob
    batch_files = glob.glob(f"similarity_pairs_{SUBSET_LANG}_batch_*.xlsx")
    for batch_file in sorted(batch_files):
        print(f"  - {batch_file}")
    
except Exception as e:
    print(f"Error saving consolidated results: {e}")
    print("However, individual batch files should still be available for recovery.")
    exit(1)