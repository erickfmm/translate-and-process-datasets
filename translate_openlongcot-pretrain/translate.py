
import argparse
from typing import List, Tuple, Optional


async def get_substrings(s: str) -> List[Tuple[int, str]]:
	substrings = []
	end = False
	while s != "":
		start = s.find("<")
		if end is not False:
			ss = s[:start]
			if ss != "" and len(ss.strip()) > 0:
				substrings.append((0, ss.strip()))
		end = s.find(">")
		substrings.append((1, s[start:end+1]))
		s = s[end+1:]
	return substrings

def is_int(s: str):
	try:
		int(s)
		return True
	except:
		return False


async def translate_substrings(substrings: List[Tuple[int, str]], pipe):
	new_substrings = []
	for type_, text in substrings:
		#print(f"{type_}\t{text}")
		if type_ == 0 and not is_int(text):
			translated = pipe(text)[0]['translation_text']
			new_substrings.append(translated)
			#print()
		else:
			new_substrings.append(text)
	return new_substrings

async def run_translation(model_name: str,
						  dataset_name: str,
						  split: str,
						  output_csv: str,
						  max_samples: Optional[int] = None,
						  progress_interval: int = 100) -> None:
	from transformers import pipeline
	from datasets import load_dataset
	import csv

	pipe = pipeline("translation", model=model_name)
	ds = load_dataset(dataset_name, split=split)
	total = len(ds) if max_samples is None else min(len(ds), max_samples)
	print(f"Translating up to {total} samples from {dataset_name}:{split} using {model_name}")

	with open(output_csv, "w", encoding="utf-8", newline='') as outputf:
		writerout = csv.writer(outputf)
		writerout.writerow(["translation"])
		for i, s in enumerate(ds["text"]):
			if max_samples is not None and i >= max_samples:
				break
			try:
				new_s = await translate_substrings(await get_substrings(s), pipe)
			except Exception as e:
				print(f"ERROR {i}: {e}")
				continue
			new_s_joined = "".join(new_s)
			writerout.writerow([new_s_joined])
			if (i + 1) % progress_interval == 0 or (i + 1) == total:
				print(f"Progress {i+1}/{total}")

	print(f"Done. Output saved to {output_csv}")


def build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Translate dataset text while preserving <...> tag blocks.")
	p.add_argument('--model', default='Helsinki-NLP/opus-mt-en-es', help='Translation model name')
	p.add_argument('--dataset', default='qq8933/OpenLongCoT-Pretrain', help='Dataset name (HF hub)')
	p.add_argument('--split', default='train', help='Dataset split')
	p.add_argument('--output-csv', default='openlong_cot_es.csv', help='Output CSV filename')
	p.add_argument('--max-samples', type=int, default=None, help='Limit number of samples (debug)')
	p.add_argument('--progress-interval', type=int, default=100, help='How often to report progress')
	return p


async def main():
	parser = build_arg_parser()
	args = parser.parse_args()
	await run_translation(model_name=args.model,
						  dataset_name=args.dataset,
						  split=args.split,
						  output_csv=args.output_csv,
						  max_samples=args.max_samples,
						  progress_interval=args.progress_interval)


if __name__ == '__main__':
	import asyncio
	asyncio.run(main())