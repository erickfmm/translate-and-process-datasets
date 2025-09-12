import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
from datasets import load_dataset
from transformers import pipeline


def configure_cache(base: Path):
	os.environ['HF_HOME'] = str(base / '.cache')
	os.environ['HUGGINGFACE_HUB_CACHE'] = str(base / '.cache')
	os.environ['TRANSFORMERS_CACHE'] = str(base / '.cache')
	os.environ['HF_DATASETS_CACHE'] = str(base / '.cache')


def translate_pairs(skip_rows: int,
					max_rows: Optional[int],
					output_excel: str,
					log_file: str,
					model_name: str,
					dataset_name: str = "embedding-data/PAQ_pairs") -> None:
	configure_cache(Path.cwd())
	pipe = pipeline("translation", model=model_name)
	dataset = load_dataset(dataset_name, streaming=False, split="train")

	all_data = []
	index = []
	processed = 0
	with open(log_file, 'w', encoding='utf-8') as f_log:
		f_log.write("time,delta,item,event\n")
		for i, data in enumerate(dataset):
			if i < skip_rows:
				continue
			if max_rows is not None and processed >= max_rows:
				break
			start_time = datetime.now()
			try:
				Q_original = data["set"][0]
				A_original = data["set"][1]
				f_log.write(f"{datetime.now()},{datetime.now()-start_time},{i},A procesar\n")
				Q_traducida = pipe(Q_original)[0]['translation_text']
				A_traducida = pipe(A_original)[0]['translation_text']
				d = {
					"Q_original": Q_original,
					"A_original": A_original,
					"Q_traducida": Q_traducida,
					"A_traducida": A_traducida
				}
				index.append(i)
				all_data.append(d)
				df = pd.DataFrame(all_data)
				df.index = index
				df.to_excel(excel_writer=output_excel, sheet_name="Hoja1")
				f_log.write(f"{datetime.now()},{datetime.now()-start_time},{i},Procesado\n")
				f_log.write(f"{datetime.now()},{datetime.now()-start_time},{i},Guardado\n")
				processed += 1
				if processed % 50 == 0:
					print(f"Processed {processed} rows (dataset index {i})")
			except Exception as e:
				f_log.write(f"{datetime.now()},{datetime.now()-start_time},{i},Error: {e}\n")
				print(f"Error at index {i}: {e}")
		f_log.write(f"{datetime.now()},0,{i},Terminó\n")
	print(f"Completed. Translated {processed} pairs -> {output_excel}")


def build_arg_parser():
	p = argparse.ArgumentParser(description="Translate PAQ question-answer pairs to Spanish.")
	p.add_argument('--skip-rows', type=int, default=0, help='Number of initial rows to skip (resume)')
	p.add_argument('--max-rows', type=int, default=None, help='Limit rows to translate (debug)')
	p.add_argument('--output-excel', default='dataset_paq_traducido.xlsx', help='Output Excel file path')
	p.add_argument('--log-file', default='log.txt', help='Log file path')
	p.add_argument('--model', default='Helsinki-NLP/opus-mt-en-es', help='Translation model name')
	p.add_argument('--dataset', default='embedding-data/PAQ_pairs', help='Source dataset name')
	return p


def main():
	parser = build_arg_parser()
	args = parser.parse_args()
	translate_pairs(skip_rows=args.skip_rows,
					max_rows=args.max_rows,
					output_excel=args.output_excel,
					log_file=args.log_file,
					model_name=args.model,
					dataset_name=args.dataset)


if __name__ == '__main__':
	main()
