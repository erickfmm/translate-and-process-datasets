# Translate OpenO1-SFT Dataset (EN -> ES)

Processes and partially translates the `O1-OPEN/OpenO1-SFT` and `O1-OPEN/OpenO1-SFT-Pro` JSONL datasets into Spanish focusing on:
- Prompt (`prompt`)
- Thought traces inside `<Thought>...</Thought>` tags
- Output text inside `<Output>...</Output>` tags

Writes an interim CSV (`coto1.csv`) and a final aggregated CSV `translated_dataset_cot.csv` with columns:
`prompt,response_thoughts,response_salida,original_prompt,original_response`

## Pipeline Summary
1. Load two JSONL files via `datasets.load_dataset` (HF filesystem URLs).
2. Normalize column names (`instruction` -> `prompt`, `output` -> `response`).
3. Concatenate into one dataset.
4. Use `Helsinki-NLP/opus-mt-en-es` translation pipeline (CPU by default; change `device`).
5. Extract `<Thought>` and `<Output>` blocks with regex.
6. Sentence tokenize thoughts/outputs (NLTK) and split long sentences (>470 tokens) to respect model limits.
7. Translate segments, join them, and record both translated and original fields.

## Requirements
- `datasets`
- `transformers`
- `nltk`
- `pandas`
- `torch` (CPU or GPU)

Install (example):
```bash
pip install datasets transformers nltk pandas torch sentencepiece sacremoses
python -c "import nltk; nltk.download('punkt')"
```

## Usage
```bash
python translate.py
```
Outputs:
- `coto1.csv` (incremental writes during processing)
- `translated_dataset_cot.csv` (final DataFrame)

### CLI Options
Run `python translate.py --help` to view all flags.

| Flag | Description | Default |
|------|-------------|---------|
| `--data-files` | One or more JSONL HF paths or local files | Two OpenO1 paths |
| `--model` | Translation model | `Helsinki-NLP/opus-mt-en-es` |
| `--device` | `auto`, `-1`, `cpu`, or CUDA index | `auto` |
| `--max-samples` | Limit number of samples processed | None |
| `--interim-file` | Interim incremental CSV | `coto1.csv` |
| `--final-csv` | Final aggregated CSV | `translated_dataset_cot.csv` |

Example (GPU 0, first 100 samples):
```bash
python translate.py --device 0 --max-samples 100 --final-csv subset_openo1.csv
```

## Configuration Points
| Setting | Location | Notes |
|---------|----------|-------|
| Device (CPU/GPU) | `device` variable near top | Set to integer GPU id if CUDA available. |
| Max segment length split | Hardcoded 470 tokens | Adjust if translation model changes. |
| Input file paths | `data_files` list | Replace with local or different HF URIs. |

## Limitations / Known Issues
- Some variable reuse (e.g., using `thought` in output loop) could be refactored.
- No robust error classification; errors are printed and skipped.
- Regex assumes non-nested identical tags.
- Long processing time for full datasets (consider batching with dataset.map + batch_size).

## Possible Improvements
- Batch translation of sentences instead of per-sentence calls.
- Add CLI args (argparse) for paths and device.
- Add logging to file with timestamps & error counts.
- Unit tests for `extract_text_between_tags` and `is_english`.

## License
See root `LICENSE`.
