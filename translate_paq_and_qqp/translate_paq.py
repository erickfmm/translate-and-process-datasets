import os
import signal
import argparse
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from datetime import datetime
from typing import Optional
from datasets import load_dataset
from openpyxl import Workbook
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class AsyncExcelWriter:
	def __init__(self, output_excel: str, flush_every: int = 5, flush_interval_seconds: float = 5.0):
		if flush_every < 1:
			raise ValueError('flush_every must be at least 1.')
		if flush_interval_seconds <= 0:
			raise ValueError('flush_interval_seconds must be greater than 0.')

		self.output_path = Path(output_excel)
		self.temp_path = self.output_path.with_name(
			f"{self.output_path.stem}.tmp{self.output_path.suffix}"
		)
		self.flush_every = flush_every
		self.flush_interval_seconds = flush_interval_seconds
		self._queue = Queue()
		self._stop_event = threading.Event()
		self._sentinel = object()
		self._error: Optional[Exception] = None
		self._pending_rows = 0
		self.saved_rows = 0
		self._last_flush_time = time.monotonic()
		self._workbook = Workbook()
		self._sheet = self._workbook.active
		self._sheet.title = 'Hoja1'
		self._sheet.append(['index', 'Q_original', 'A_original', 'Q_traducida', 'A_traducida'])
		self._thread = threading.Thread(target=self._run, name='xlsx-writer', daemon=False)

	def start(self) -> None:
		self._thread.start()

	def submit(self, dataset_index: int, row: dict) -> None:
		self.raise_if_failed()
		self._queue.put((dataset_index, row.copy()))

	def close(self) -> None:
		self._stop_event.set()
		self._queue.put(self._sentinel)
		self._thread.join()
		self.raise_if_failed()

	def raise_if_failed(self) -> None:
		if self._error is not None:
			raise RuntimeError('The XLSX writer thread failed.') from self._error

	def _run(self) -> None:
		try:
			while not self._stop_event.is_set() or not self._queue.empty():
				should_flush = False
				try:
					item = self._queue.get(timeout=0.5)
				except Empty:
					item = None

				if item is self._sentinel:
					continue

				if item is not None:
					dataset_index, row = item
					self._sheet.append([
						dataset_index,
						row['Q_original'],
						row['A_original'],
						row['Q_traducida'],
						row['A_traducida'],
					])
					self._pending_rows += 1
					should_flush = self._pending_rows >= self.flush_every

				if self._pending_rows > 0 and (
					time.monotonic() - self._last_flush_time >= self.flush_interval_seconds
				):
					should_flush = True

				if should_flush:
					self._flush()

			if self._pending_rows > 0:
				self._flush()
		except Exception as exc:
			self._error = exc
			self._stop_event.set()

	def _flush(self) -> None:
		self.output_path.parent.mkdir(parents=True, exist_ok=True)
		try:
			self._workbook.save(self.temp_path)
			os.replace(self.temp_path, self.output_path)
		except Exception:
			if self.temp_path.exists():
				self.temp_path.unlink()
			raise
		self.saved_rows += self._pending_rows
		self._pending_rows = 0
		self._last_flush_time = time.monotonic()


def configure_cache(base: Path):
	os.environ['HF_HOME'] = str(base / '.cache')
	os.environ['HUGGINGFACE_HUB_CACHE'] = str(base / '.cache')
	os.environ['TRANSFORMERS_CACHE'] = str(base / '.cache')
	os.environ['HF_DATASETS_CACHE'] = str(base / '.cache')


def resolve_device(device: Optional[str]) -> str:
	if device is None:
		return 'cuda' if torch.cuda.is_available() else 'cpu'
	try:
		resolved_device = str(torch.device(device))
	except (TypeError, RuntimeError) as exc:
		raise ValueError(
			f"Invalid device '{device}'. Use values like 'cpu', 'cuda', or 'cuda:0'."
		) from exc
	if resolved_device.startswith('cuda') and not torch.cuda.is_available():
		raise RuntimeError('CUDA device requested, but CUDA is not available in this environment.')
	return resolved_device


def load_translation_model(model_name: str, device: Optional[str]):
	resolved_device = resolve_device(device)
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
	model.to(resolved_device)
	model.eval()
	return tokenizer, model, resolved_device


@torch.inference_mode()
def translate_texts(texts, tokenizer, model, device: str):
	encoded = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
	encoded = {key: value.to(device) for key, value in encoded.items()}
	generated_tokens = model.generate(**encoded)
	return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def install_stop_handlers(stop_event: threading.Event):
	previous_handlers = {}

	def handle_stop(signum, _frame):
		if not stop_event.is_set():
			print(f"\\nReceived signal {signum}. Stopping after the current item and flushing pending XLSX rows...")
		stop_event.set()

	for signum in (signal.SIGINT, signal.SIGTERM):
		previous_handlers[signum] = signal.getsignal(signum)
		signal.signal(signum, handle_stop)
	return previous_handlers


def restore_signal_handlers(previous_handlers):
	for signum, handler in previous_handlers.items():
		signal.signal(signum, handler)


def translate_pairs(skip_rows: int,
					max_rows: Optional[int],
					output_excel: str,
					log_file: str,
					model_name: str,
					device: Optional[str] = None,
					flush_every: int = 5,
					flush_interval_seconds: float = 5.0,
					dataset_name: str = "embedding-data/PAQ_pairs") -> None:
	configure_cache(Path.cwd())
	tokenizer, model, resolved_device = load_translation_model(model_name, device)
	dataset = load_dataset(dataset_name, streaming=False, split="train")
	writer = AsyncExcelWriter(
		output_excel=output_excel,
		flush_every=flush_every,
		flush_interval_seconds=flush_interval_seconds,
	)
	stop_requested = threading.Event()
	previous_handlers = install_stop_handlers(stop_requested)
	writer.start()

	processed = 0
	last_index = None
	f_log = open(log_file, 'w', encoding='utf-8', buffering=1)
	try:
		f_log.write("time,delta,item,event\n")
		for i, data in enumerate(dataset):
			last_index = i
			writer.raise_if_failed()
			if stop_requested.is_set():
				f_log.write(f"{datetime.now()},0,{i},Detención solicitada\n")
				break
			if i < skip_rows:
				continue
			if max_rows is not None and processed >= max_rows:
				break
			start_time = datetime.now()
			try:
				Q_original = data["set"][0]
				A_original = data["set"][1]
				f_log.write(f"{datetime.now()},{datetime.now()-start_time},{i},A procesar\n")
				Q_traducida, A_traducida = translate_texts(
					[Q_original, A_original], tokenizer, model, resolved_device
				)
				d = {
					"Q_original": Q_original,
					"A_original": A_original,
					"Q_traducida": Q_traducida,
					"A_traducida": A_traducida
				}
				f_log.write(f"{datetime.now()},{datetime.now()-start_time},{i},Procesado\n")
				writer.submit(i, d)
				f_log.write(f"{datetime.now()},{datetime.now()-start_time},{i},Encolado para guardado\n")
				processed += 1
				if processed % 50 == 0:
					print(
						f"Processed {processed} rows (dataset index {i}, flushed {writer.saved_rows} rows to disk)"
					)
			except Exception as e:
				f_log.write(f"{datetime.now()},{datetime.now()-start_time},{i},Error: {e}\n")
				print(f"Error at index {i}: {e}")
	finally:
		final_item = last_index if last_index is not None else -1
		try:
			writer.close()
		except Exception as exc:
			f_log.write(f"{datetime.now()},0,{final_item},Error al sincronizar XLSX: {exc}\n")
			raise
		else:
			f_log.write(f"{datetime.now()},0,{final_item},XLSX sincronizado\n")
		finally:
			f_log.write(f"{datetime.now()},0,{final_item},Terminó\n")
			f_log.close()
			restore_signal_handlers(previous_handlers)
	print(
		f"Completed. Translated {processed} pairs -> {output_excel} "
		f"(flushed {writer.saved_rows} rows to disk)"
	)


def build_arg_parser():
	p = argparse.ArgumentParser(description="Translate PAQ question-answer pairs to Spanish.")
	p.add_argument('--skip-rows', type=int, default=0, help='Number of initial rows to skip (resume)')
	p.add_argument('--max-rows', type=int, default=None, help='Limit rows to translate (debug)')
	p.add_argument('--output-excel', default='dataset_paq_traducido.xlsx', help='Output Excel file path')
	p.add_argument('--log-file', default='log.txt', help='Log file path')
	p.add_argument('--model', default='Helsinki-NLP/opus-mt-en-es', help='Translation model name')
	p.add_argument('--device', default=None, help='Device for inference (e.g. cpu, cuda, cuda:0)')
	p.add_argument('--flush-every', type=int, default=5, help='Queue this many translated rows before forcing an XLSX flush')
	p.add_argument('--flush-interval-seconds', type=float, default=5.0, help='Maximum seconds between XLSX flushes')
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
					device=args.device,
					flush_every=args.flush_every,
					flush_interval_seconds=args.flush_interval_seconds,
					dataset_name=args.dataset)


if __name__ == '__main__':
	main()
