# translate-and-process-datasets

Small collection of focused scripts to translate or process several open datasets (English → Spanish, filtering, exporting, and syncing to Hugging Face Hub). Each subfolder has its own detailed `README.md` with usage instructions.

## Repository Structure

| Folder | Purpose |
|--------|---------|
| `filter_no_chinese_k12_resample/` | Filter out rows containing Chinese characters in images using a YOLO model, export to CSV/PNG, push filtered dataset to Hub. |
| `translate_openlongcot-pretrain/` | Translate `qq8933/OpenLongCoT-Pretrain` dataset while preserving XML-like tags. |
| `translate_openo1-sft/` | Extract and translate prompts, <Thought> and <Output> segments from OpenO1-SFT datasets. |
| `translate_paq_and_qqp/` | Translate PAQ QA pairs and QQP triplets with incremental Excel + logging. |

## Quick Start (Generic)
Create and activate a virtual environment, then install needed packages per subfolder.

```bash
python -m venv .venv
./.venv/Scripts/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install --upgrade pip
# Example for PAQ/QQP scripts:
pip install -r translate_paq_and_qqp/requirements.txt
```

Run a script (example):
```bash
python translate_openlongcot-pretrain/translate.py
```

All scripts now support `--help` for CLI options, e.g.:
```bash
python filter_no_chinese_k12_resample/filter_chinese.py --help
python translate_paq_and_qqp/translate_paq.py --skip-rows 5000 --max-rows 1000 --output-excel paq_partial.xlsx
```

See each subfolder for: dependencies, configuration variables, outputs, and troubleshooting tips.

## Hugging Face Usage Tips
- Login once before pushing datasets: `huggingface-cli login`
- Use local caching env vars if handling very large datasets.

## License
See `LICENSE` in the root of this repository.

## Contributions
Feel free to open issues or PRs for improvements (batching, CLI args, robustness, additional language pairs).
