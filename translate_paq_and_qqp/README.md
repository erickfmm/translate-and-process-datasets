# Translate PAQ and QQP Datasets (EN -> ES)

Scripts to translate large-scale question-answer and query similarity datasets into Spanish:
- `embedding-data/PAQ_pairs` (Q/A pairs)
- `embedding-data/QQP_triplets` (Query + positive + negative examples)

Both use the OPUS MT model `Helsinki-NLP/opus-mt-en-es` via `transformers` pipeline and export incremental results to Excel plus logging.

## Contents
| Script | Purpose |
|--------|---------|
| `translate_paq.py` | Iteratively translates PAQ Q/A pairs, writing to `dataset_paq_traducido.xlsx` and `log.txt`. |
| `translate_qqp.py` | Translates QQP query, positive, and negatives to `dataset_qqp_traducido.xlsx` and logs progress. |
| `requirements.txt` | Dependencies for these scripts. |

## Requirements
See `requirements.txt`:
- `datasets`, `transformers`, `pandas`, `openpyxl`, `torch`, `sentencepiece`, `sacremoses`, `hf_xet`, `huggingface_hub[hf_xet]`

Install:
```bash
pip install -r requirements.txt
```

## Environment & Caching
Scripts set local cache directories (`.cache` in working dir) for model + dataset artifacts via environment variables.

## Usage Examples
PAQ (adjust `skip_n_rows` to resume partial translation):
```bash
python translate_paq.py
```
QQP (adjust `skip_n_rows` similarly):
```bash
python translate_qqp.py
```

### CLI Options

Both scripts expose flags (run `--help`). Shared concepts: skipping initial rows to resume, limiting rows for debugging, and customizing output/log filenames.

#### `translate_paq.py`
| Flag | Description | Default |
|------|-------------|---------|
| `--skip-rows` | Skip this many initial dataset rows (resume) | 0 |
| `--max-rows` | Limit number of rows to translate | None |
| `--output-excel` | Output Excel file | `dataset_paq_traducido.xlsx` |
| `--log-file` | Log file path | `log.txt` |
| `--model` | Translation model | `Helsinki-NLP/opus-mt-en-es` |
| `--dataset` | Source dataset name | `embedding-data/PAQ_pairs` |

Example (resume at 6500, translate 500 rows):
```bash
python translate_paq.py --skip-rows 6500 --max-rows 500 --output-excel paq_6500_7000.xlsx
```

#### `translate_qqp.py`
| Flag | Description | Default |
|------|-------------|---------|
| `--skip-rows` | Skip this many initial rows | 0 |
| `--max-rows` | Limit rows to translate | None |
| `--output-excel` | Output Excel file | `dataset_qqp_traducido.xlsx` |
| `--log-file` | Log file path | `log.txt` |
| `--model` | Translation model | `Helsinki-NLP/opus-mt-en-es` |
| `--dataset` | Source dataset name | `embedding-data/QQP_triplets` |

Example (first 120 rows only):
```bash
python translate_qqp.py --max-rows 120 --output-excel qqp_first120.xlsx
```

## Output Artifacts
| File | Description |
|------|-------------|
| `dataset_paq_traducido.xlsx` | Accumulated translated PAQ rows. |
| `dataset_qqp_traducido.xlsx` | Accumulated translated QQP rows. |
| `log.txt` | CSV-like log: timestamp, elapsed delta, row index, event. |

## Performance Tips
- Consider GPU: set `device` in a custom pipeline if large throughput needed.
- Batch translations: current implementation translates each string individually.
- Periodically archive the Excel file to avoid corruption if interrupted.

## Resuming Work
- Adjust `skip_n_rows` to the last successfully translated index + 1.
- Ensure previous outputs remain in place (script rewrites entire Excel each loop).

## Potential Improvements
- Replace per-row DataFrame write with buffered writes or Parquet.
- Add checkpointing & safe resume logic.
- Use batched pipeline calls to reduce overhead.
- Optionally output CSV instead of Excel for speed.

## License
See root `LICENSE`.
