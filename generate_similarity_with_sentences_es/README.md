# Sentence Similarity Generation with Spanish Embeddings

This script generates cosine similarity pairs for Spanish sentences using sentence transformers.

## Features

- Loads Spanish sentences from the `agentlans/multilingual-sentences` dataset
- Uses Spanish sentence similarity model: `hiiamsid/sentence_similarity_spanish_es`
- Generates targeted similarity comparisons based on configurable parameters
- Processes data in batches for memory efficiency
- Outputs results to Excel format

## Configuration

Key parameters that control the similarity generation:

- `N_REPETITIONS_PER_ROW`: Number of times each row is compared to other rows (default: 5)
- `N_ROWS_PER_ROW`: Number of other rows to compare each row to (default: 5)
- `N_ROWS_PER_BATCH`: Number of rows processed in each batch (default: 1000)

## How it works

1. **Dataset Loading**: Loads Spanish sentences from the multilingual sentences dataset
2. **Batch Processing**: Processes sentences in batches to manage memory usage
3. **Targeted Comparisons**: For each sentence:
   - Repeats comparison N_REPETITIONS_PER_ROW times
   - Randomly selects N_ROWS_PER_ROW other sentences for comparison
   - Computes cosine similarity using sentence transformers
4. **Output**: Saves all similarity pairs to an Excel file with columns:
   - `idx1`, `idx2`: Original indices of the sentences
   - `sentence1`, `sentence2`: The actual sentences being compared
   - `similarity`: Cosine similarity score (0-1)

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```bash
python generate_similarity.py
```

## Output

The script generates a file named `similarity_pairs_es.xlsx` containing all the computed similarity pairs.

## Authentication

You may need to login to Hugging Face to access the dataset:
```bash
huggingface-cli login
```