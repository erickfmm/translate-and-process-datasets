# Translate OpenLongCoT Pretrain

Asynchronously translates the `qq8933/OpenLongCoT-Pretrain` dataset from English to Spanish while preserving XML‑like tagged substrings (segments enclosed in `<...>`).

## How It Works
1. Loads dataset split `train` via `datasets.load_dataset`.
2. Splits each text into alternating plain segments and tag tokens using `get_substrings`.
3. Translates only plain (non‑integer) segments using the `Helsinki-NLP/opus-mt-en-es` translation pipeline.
4. Reassembles the sequence and writes each translated line to `openlong_cot_es.csv` (one column, one row per original sample).

## Requirements
- `datasets`
- `transformers`
- Model weights auto-downloaded (`Helsinki-NLP/opus-mt-en-es`)

Install:
```bash
pip install datasets transformers sentencepiece sacremoses
```

## Usage
```bash
python translate.py
```
Output file: `openlong_cot_es.csv` in the working directory.

### CLI Options
Run `python translate.py --help` for the full list.

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Translation model name | `Helsinki-NLP/opus-mt-en-es` |
| `--dataset` | HF dataset name | `qq8933/OpenLongCoT-Pretrain` |
| `--split` | Dataset split | `train` |
| `--output-csv` | Output CSV filename | `openlong_cot_es.csv` |
| `--max-samples` | Limit number of samples (debug) | None |
| `--progress-interval` | Rows between progress logs | 100 |

Example limiting to 200 samples:
```bash
python translate.py --max-samples 200 --output-csv sample_openlongcot.csv
```

## Customization
| Need | Where to change |
|------|-----------------|
| Different source model | Change model name in `pipeline("translation", model=...)`. |
| Different language pair | Pick another OPUS MT model, e.g. `opus-mt-en-fr`. |
| Preserve extra tokens | Adjust logic in `get_substrings` / filter condition in `translate_substrings`. |
| Batch performance | Wrap multiple strings and call pipeline in batches (current code translates per segment). |

## Caveats
- The parsing logic assumes tags use `<` and `>` and are not nested irregularly.
- Large inputs with many tags can increase API/model calls (one per plain segment).
- Script runs synchronously inside the event loop even though functions are `async` (fine for this use).

## Future Improvements (Ideas)
- Batch translation of multiple segments for speed.
- Add progress bar and error logging file.
- Preserve original text alongside translation (2-column CSV).

## License
See root `LICENSE`.
