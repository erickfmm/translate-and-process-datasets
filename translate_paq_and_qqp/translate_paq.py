import os
import signal
import argparse
import multiprocessing as mp
import queue as _queue
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
from datasets import load_dataset
from openpyxl import Workbook

from translation_engine import (
	make_engine,
	engine_config_from_args,
	add_engine_args,
	RetryExhaustedError,
)
from gpu_temp_guard import TempGuard


# ---------------------------------------------------------------------------
# Message type constants for inter-process communication
# ---------------------------------------------------------------------------
MSG_BATCH_RESULT = 'batch_result'
MSG_WORKER_READY = 'worker_ready'
MSG_WORKER_ERROR = 'worker_error'
MSG_WORKER_DONE = 'worker_done'


def configure_cache(base: Path):
	os.environ['HF_HOME'] = str(base / '.cache')
	os.environ['HUGGINGFACE_HUB_CACHE'] = str(base / '.cache')
	os.environ['TRANSFORMERS_CACHE'] = str(base / '.cache')
	os.environ['HF_DATASETS_CACHE'] = str(base / '.cache')


# ---------------------------------------------------------------------------
# Worker (slave) process
# ---------------------------------------------------------------------------
def worker_process(
	worker_id: int,
	task_queue: mp.Queue,
	result_queue: mp.Queue,
	engine_config: Dict[str, Any],
	offload_event: Optional[Any] = None,
):
	"""
	Slave worker process. Loads the translation engine and waits for batches
	from the master via *task_queue*. Each batch is a list of
	(dataset_index, Q_original, A_original) tuples.
	Results are sent back through *result_queue*.

	If *offload_event* (an ``multiprocessing.Event``) is provided, the worker
	checks it before fetching each new task: when set, it offloads the model
	VRAM -> system RAM and blocks until cleared, then reloads to GPU.
	"""
	try:
		configure_cache(Path.cwd())
		engine = make_engine(**engine_config)
		result_queue.put((MSG_WORKER_READY, worker_id, None))

		while True:
			# Honor temperature offload: move weights to RAM while idle so
			# the GPU can cool. Reload to GPU before doing any work.
			if offload_event is not None and offload_event.is_set():
				engine.offload_to_cpu()
				# Spin until cleared; re-check the queue non-blockingly so a
				# None sentinel (shutdown) is not missed.
				while offload_event.is_set():
					try:
						task = task_queue.get(timeout=1)
					except _queue.Empty:
						continue
					if task is None:
						result_queue.put((MSG_WORKER_DONE, worker_id, None))
						return
				engine.reload_to_device()

			task = task_queue.get()
			if task is None:
				# Sentinel: no more work
				result_queue.put((MSG_WORKER_DONE, worker_id, None))
				break

			batch: List[Tuple[int, str, str]] = task
			results: List[Dict[str, Any]] = []

			for dataset_index, Q_original, A_original in batch:
				Q_traducida, A_traducida = engine.translate([Q_original, A_original])
				results.append({
					'index': dataset_index,
					'Q_original': Q_original,
					'A_original': A_original,
					'Q_traducida': Q_traducida,
					'A_traducida': A_traducida,
				})

			result_queue.put((MSG_BATCH_RESULT, worker_id, results))

	except Exception as exc:
		result_queue.put((MSG_WORKER_ERROR, worker_id, exc))


# ---------------------------------------------------------------------------
# Master coordinator
# ---------------------------------------------------------------------------
class MasterCoordinator:
	"""
	Master node that:
	1. Loads the dataset and creates batches of rows.
	2. Dispatches batches to worker processes via per-worker task queues.
	3. Collects results and reorders them by original dataset index.
	4. Writes ordered results to a single XLSX file.
	"""

	def __init__(
		self,
		output_excel: str,
		log_file: str,
		engine_config: Dict[str, Any],
		num_workers: int,
		batch_size: int,
		skip_rows: int,
		max_rows: Optional[int],
		flush_every: int,
		flush_interval_seconds: float,
		dataset_name: str,
		temp_guard: Optional[TempGuard] = None,
		offload_enabled: bool = False,
	):
		self.output_excel = output_excel
		self.log_file = log_file
		self.engine_config = engine_config
		self.num_workers = num_workers
		self.batch_size = batch_size
		self.skip_rows = skip_rows
		self.max_rows = max_rows
		self.flush_every = flush_every
		self.flush_interval_seconds = flush_interval_seconds
		self.dataset_name = dataset_name
		self.temp_guard = temp_guard
		self.offload_enabled = offload_enabled

		# Results bookkeeping
		self._results_buffer: Dict[int, Dict[str, Any]] = {}
		self._next_write_index: int = skip_rows

		# XLSX writer state
		self._workbook: Optional[Workbook] = None
		self._sheet = None
		self._pending_rows: int = 0
		self._saved_rows: int = 0
		self._last_flush_time: float = time.monotonic()
		self._temp_path: Optional[Path] = None
		self._output_path: Optional[Path] = None

	def run(self) -> None:
		configure_cache(Path.cwd())
		dataset = load_dataset(self.dataset_name, streaming=False, split="train")
		self._output_path = Path(self.output_excel)
		self._temp_path = self._output_path.with_name(
			f"{self._output_path.stem}.tmp{self._output_path.suffix}"
		)

		# Prepare XLSX workbook
		self._workbook = Workbook()
		self._sheet = self._workbook.active
		self._sheet.title = 'Hoja1'
		self._sheet.append(['index', 'Q_original', 'A_original', 'Q_traducida', 'A_traducida'])

		# Build batches from the dataset
		batches: List[List[Tuple[int, str, str]]] = []
		current_batch: List[Tuple[int, str, str]] = []
		processed = 0

		for i, data in enumerate(dataset):
			if i < self.skip_rows:
				continue
			if self.max_rows is not None and processed >= self.max_rows:
				break
			Q_original = data["set"][0]
			A_original = data["set"][1]
			current_batch.append((i, Q_original, A_original))
			processed += 1
			if len(current_batch) >= self.batch_size:
				batches.append(current_batch)
				current_batch = []
		if current_batch:
			batches.append(current_batch)

		total_rows = processed
		total_batches = len(batches)
		print(f"Dataset loaded: {total_rows} rows in {total_batches} batches "
			  f"(batch_size={self.batch_size}, workers={self.num_workers})")

		# Multiprocessing infrastructure
		ctx = mp.get_context('spawn')
		task_queues: List[mp.Queue] = []
		result_queue: mp.Queue = ctx.Queue()

		# Shared offload signal: master mirrors the TempGuard's offload_event
		# here so worker processes can offload their model VRAM -> RAM.
		offload_mp_event = None
		if self.temp_guard is not None and self.offload_enabled:
			offload_mp_event = ctx.Event()

		# Start worker processes
		workers: List[mp.Process] = []
		for wid in range(self.num_workers):
			tq = ctx.Queue()
			task_queues.append(tq)
			p = ctx.Process(
				target=worker_process,
				args=(wid, tq, result_queue, self.engine_config, offload_mp_event),
				name=f'worker-{wid}',
				daemon=True,
			)
			workers.append(p)
			p.start()

		print(f"Started {self.num_workers} worker processes, waiting for engine to load...")

		# Wait for all workers to signal readiness
		ready_workers = set()
		while len(ready_workers) < self.num_workers:
			msg_type, wid, payload = result_queue.get()
			if msg_type == MSG_WORKER_READY:
				ready_workers.add(wid)
				print(f"  Worker {wid} ready.")
			elif msg_type == MSG_WORKER_ERROR:
				raise RuntimeError(f"Worker {wid} failed during startup: {payload}")

		print("All workers ready. Distributing batches...")

		# Open log file
		f_log = open(self.log_file, 'w', encoding='utf-8', buffering=1)
		f_log.write("time,delta,item,event\n")

		# Start temperature guard if present
		if self.temp_guard is not None:
			self.temp_guard._log = lambda m: (
				print(m) or f_log.write(f"{datetime.now()},0,-,{m}\n")
			)
			self.temp_guard.start()
			offload_msg = (
				"on (VRAM<->RAM shuffle)" if self.offload_enabled else "off (idle only)"
			)
			print(
				f"TempGuard active: pause>={self.temp_guard.temp_max}C "
				f"resume<={self.temp_guard.temp_resume}C "
				f"stop>={self.temp_guard.temp_stop}C "
				f"(every {self.temp_guard.check_interval}s, offload {offload_msg})"
			)

		stop_requested = False

		def handle_stop(signum, _frame):
			nonlocal stop_requested
			if not stop_requested:
				print(f"\nReceived signal {signum}. Stopping after current batches...")
			stop_requested = True

		previous_handlers = {}
		for signum in (signal.SIGINT, signal.SIGTERM):
			previous_handlers[signum] = signal.getsignal(signum)
			signal.signal(signum, handle_stop)

		# Dispatch initial batches: one per worker
		batch_idx = 0
		active_workers = set(range(self.num_workers))

		for wid in range(self.num_workers):
			if batch_idx < total_batches and not stop_requested:
				task_queues[wid].put(batches[batch_idx])
				batch_idx += 1

		batches_done = 0
		kill_triggered = False

		try:
			while batches_done < total_batches and not stop_requested:
				# Critical temperature: hard-kill all workers and exit.
				if (
					self.temp_guard is not None
					and self.temp_guard.kill_event.is_set()
				):
					kill_triggered = True
					break

				# Mirror the guard's offload request into the shared
				# multiprocessing.Event so workers move their model VRAM->RAM
				# while the GPU is meant to cool down.
				if offload_mp_event is not None:
					if self.temp_guard.offload_event.is_set():
						offload_mp_event.set()
					else:
						offload_mp_event.clear()

				# Block waiting for a result from any worker. When the
				# temperature guard is active, poll with a timeout so the
				# master can react to kill_event even while workers are
				# idle/paused.
				if self.temp_guard is not None:
					try:
						msg_type, wid, payload = result_queue.get(
							timeout=self.temp_guard.check_interval
						)
					except _queue.Empty:
						continue
				else:
					msg_type, wid, payload = result_queue.get()

				if msg_type == MSG_WORKER_ERROR:
					print(f"Worker {wid} crashed: {payload}")
					f_log.write(f"{datetime.now()},0,-,Worker {wid} crashed: {payload}\n")
					active_workers.discard(wid)
					# Retry exhaustion is fatal -> stop the whole pipeline.
					# Other transient worker crashes just retire the worker.
					if isinstance(payload, RetryExhaustedError) or not active_workers:
						raise RuntimeError(
							f"Stopping pipeline due to worker {wid} error: {payload}"
						)
					continue

				if msg_type == MSG_WORKER_DONE:
					active_workers.discard(wid)
					continue

				if msg_type != MSG_BATCH_RESULT:
					continue

				# Process batch results
				results: List[Dict[str, Any]] = payload
				for r in results:
					idx = r['index']
					self._results_buffer[idx] = r
					f_log.write(f"{datetime.now()},0,{idx},Procesado por worker {wid}\n")

				batches_done += 1

				# Flush ordered results to XLSX
				self._flush_ordered()

				if batches_done % 5 == 0 or batches_done == total_batches:
					print(
						f"Progress: {batches_done}/{total_batches} batches done, "
						f"{self._saved_rows} rows written to XLSX"
					)

				# Send next batch to this worker if available
				if batch_idx < total_batches and not stop_requested and wid in active_workers:
					# Honor temperature pause: don't dispatch new work,
					# in-flight batches finish and the GPU cools down.
					if (
						self.temp_guard is not None
						and self.temp_guard.pause_event.is_set()
					):
						continue
					task_queues[wid].put(batches[batch_idx])
					batch_idx += 1

			if kill_triggered:
				# Temperature stop threshold reached: SIGKILL every worker
				# and exit immediately without flushing.
				f_log.write(
					f"{datetime.now()},0,-,TempGuard STOP triggered; "
					f"hard-killing workers\n"
				)
				f_log.flush()
				for p in workers:
					if p.is_alive():
						p.kill()
						p.join(timeout=3)
				os._exit(2)

			# Send stop sentinels to all active workers
			for wid in range(self.num_workers):
				if wid in active_workers:
					task_queues[wid].put(None)

			# Drain remaining results from workers
			while batches_done < total_batches and active_workers:
				try:
					msg_type, wid, payload = result_queue.get(timeout=10.0)
				except Exception:
					break

				if msg_type == MSG_WORKER_DONE or msg_type == MSG_WORKER_ERROR:
					active_workers.discard(wid)
					continue

				if msg_type == MSG_BATCH_RESULT:
					results = payload
					for r in results:
						self._results_buffer[r['index']] = r
					batches_done += 1
					self._flush_ordered()

			# Final ordered flush
			self._flush_ordered(force=True)
			self._flush_xlsx()

			f_log.write(f"{datetime.now()},0,-,XLSX sincronizado\n")
			f_log.write(f"{datetime.now()},0,-,Terminó\n")

		except Exception as exc:
			f_log.write(f"{datetime.now()},0,-,Error: {exc}\n")
			raise
		finally:
			# Terminate workers
			for p in workers:
				if p.is_alive():
					p.terminate()
					p.join(timeout=3)

			if self.temp_guard is not None:
				self.temp_guard.stop()

			f_log.close()
			for signum, handler in previous_handlers.items():
				signal.signal(signum, handler)

		print(
			f"Completed. Translated {total_rows} pairs -> {self.output_excel} "
			f"({self._saved_rows} rows written to disk)"
		)

	def _flush_ordered(self, force: bool = False) -> None:
		"""Write results in order from the buffer to the XLSX sheet."""
		while self._next_write_index in self._results_buffer:
			r = self._results_buffer.pop(self._next_write_index)
			self._sheet.append([
				r['index'],
				r['Q_original'],
				r['A_original'],
				r['Q_traducida'],
				r['A_traducida'],
			])
			self._pending_rows += 1
			self._next_write_index += 1

		# Periodic XLSX flush to disk
		if self._pending_rows > 0 and (
			force
			or self._pending_rows >= self.flush_every
			or (time.monotonic() - self._last_flush_time >= self.flush_interval_seconds)
		):
			self._flush_xlsx()

	def _flush_xlsx(self) -> None:
		if self._pending_rows == 0:
			return
		self._output_path.parent.mkdir(parents=True, exist_ok=True)
		try:
			self._workbook.save(self._temp_path)
			os.replace(self._temp_path, self._output_path)
		except Exception:
			if self._temp_path.exists():
				self._temp_path.unlink()
			raise
		self._saved_rows += self._pending_rows
		self._pending_rows = 0
		self._last_flush_time = time.monotonic()


# ---------------------------------------------------------------------------
# Single-process mode (used when --workers=1, original behaviour)
# ---------------------------------------------------------------------------
def translate_pairs_single(skip_rows: int,
						   max_rows: Optional[int],
						   output_excel: str,
						   log_file: str,
						   engine_config: Dict[str, Any],
						   flush_every: int = 5,
						   flush_interval_seconds: float = 5.0,
						   dataset_name: str = "embedding-data/PAQ_pairs",
						   temp_guard: Optional[TempGuard] = None,
						   offload_enabled: bool = False) -> None:
	"""Single-process mode with in-order buffered writing."""
	configure_cache(Path.cwd())
	engine = make_engine(**engine_config)
	dataset = load_dataset(dataset_name, streaming=False, split="train")

	output_path = Path(output_excel)
	temp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
	workbook = Workbook()
	sheet = workbook.active
	sheet.title = 'Hoja1'
	sheet.append(['index', 'Q_original', 'A_original', 'Q_traducida', 'A_traducida'])

	# Buffer for reordering
	buffer: Dict[int, dict] = {}
	next_write_index = skip_rows
	pending_rows = 0
	saved_rows = 0
	last_flush_time = time.monotonic()

	def flush_xlsx():
		nonlocal pending_rows, saved_rows, last_flush_time
		if pending_rows == 0:
			return
		output_path.parent.mkdir(parents=True, exist_ok=True)
		try:
			workbook.save(temp_path)
			os.replace(temp_path, output_path)
		except Exception:
			if temp_path.exists():
				temp_path.unlink()
			raise
		saved_rows += pending_rows
		pending_rows = 0
		last_flush_time = time.monotonic()

	def flush_ordered(force=False):
		nonlocal next_write_index, pending_rows
		while next_write_index in buffer:
			row = buffer.pop(next_write_index)
			sheet.append([
				next_write_index,
				row['Q_original'],
				row['A_original'],
				row['Q_traducida'],
				row['A_traducida'],
			])
			pending_rows += 1
			next_write_index += 1
		if pending_rows > 0 and (
			force
			or pending_rows >= flush_every
			or (time.monotonic() - last_flush_time >= flush_interval_seconds)
		):
			flush_xlsx()

	processed = 0
	last_index = None
	stop_requested = False
	kill_triggered = False
	f_log = open(log_file, 'w', encoding='utf-8', buffering=1)
	try:
		f_log.write("time,delta,item,event\n")
		if temp_guard is not None:
			temp_guard._log = lambda m: (
				print(m) or f_log.write(f"{datetime.now()},0,-,{m}\n")
			)
			temp_guard.start()
			offload_msg = (
				"on (VRAM<->RAM shuffle)" if offload_enabled else "off (idle only)"
			)
			print(
				f"TempGuard active: pause>={temp_guard.temp_max}C "
				f"resume<={temp_guard.temp_resume}C "
				f"stop>={temp_guard.temp_stop}C "
				f"(every {temp_guard.check_interval}s, offload {offload_msg})"
			)
		for i, data in enumerate(dataset):
			last_index = i
			if i < skip_rows:
				continue
			if max_rows is not None and processed >= max_rows:
				break
			# Temperature guard: wait while paused, abort if kill latched.
			if temp_guard is not None:
				if temp_guard.pause_event.is_set() and not temp_guard.kill_event.is_set():
					if offload_enabled:
						engine.offload_to_cpu()
					while (
						temp_guard.pause_event.is_set()
						and not temp_guard.kill_event.is_set()
					):
						time.sleep(1)
					if temp_guard.kill_event.is_set():
						kill_triggered = True
						break
					if offload_enabled:
						engine.reload_to_device()
				elif temp_guard.kill_event.is_set():
					kill_triggered = True
					break
			start_time = datetime.now()
			try:
				Q_original = data["set"][0]
				A_original = data["set"][1]
				f_log.write(f"{datetime.now()},{datetime.now()-start_time},{i},A procesar\n")
				Q_traducida, A_traducida = engine.translate([Q_original, A_original])
				d = {
					"Q_original": Q_original,
					"A_original": A_original,
					"Q_traducida": Q_traducida,
					"A_traducida": A_traducida
				}
				f_log.write(f"{datetime.now()},{datetime.now()-start_time},{i},Procesado\n")
				buffer[i] = d
				flush_ordered()
				f_log.write(f"{datetime.now()},{datetime.now()-start_time},{i},Encolado para guardado\n")
				processed += 1
				if processed % 50 == 0:
					print(
						f"Processed {processed} rows (dataset index {i}, flushed {saved_rows} rows to disk)"
					)
			except RetryExhaustedError as e:
				# Retries exhausted on this item: stop the pipeline cleanly.
				f_log.write(
					f"{datetime.now()},{datetime.now()-start_time},{i},"
					f"RetryExhausted: {e}\n"
				)
				print(f"Fatal at index {i}: retries exhausted: {e}")
				stop_requested = True
				break
			except Exception as e:
				f_log.write(f"{datetime.now()},{datetime.now()-start_time},{i},Error: {e}\n")
				print(f"Error at index {i}: {e}")
	finally:
		final_item = last_index if last_index is not None else -1
		try:
			flush_ordered(force=True)
			flush_xlsx()
		except Exception as exc:
			f_log.write(f"{datetime.now()},0,{final_item},Error al sincronizar XLSX: {exc}\n")
			raise
		else:
			f_log.write(f"{datetime.now()},0,{final_item},XLSX sincronizado\n")
		finally:
			if temp_guard is not None:
				temp_guard.stop()
			f_log.write(f"{datetime.now()},0,{final_item},Terminó\n")
			f_log.close()
	if kill_triggered:
		print("TempGuard STOP triggered; exiting (code 2).")
		os._exit(2)
	status = "stopped" if stop_requested else "Completed"
	print(
		f"{status}. Translated {processed} pairs -> {output_excel} "
		f"(flushed {saved_rows} rows to disk)"
	)
	if stop_requested:
		raise SystemExit(1)


def build_arg_parser():
	p = argparse.ArgumentParser(description="Translate PAQ question-answer pairs to Spanish.")
	p.add_argument('--skip-rows', type=int, default=0, help='Number of initial rows to skip (resume)')
	p.add_argument('--max-rows', type=int, default=None, help='Limit rows to translate (debug)')
	p.add_argument('--output-excel', default='dataset_paq_traducido.xlsx', help='Output Excel file path')
	p.add_argument('--log-file', default='log.txt', help='Log file path')
	p.add_argument('--flush-every', type=int, default=5, help='Queue this many translated rows before forcing an XLSX flush')
	p.add_argument('--flush-interval-seconds', type=float, default=5.0, help='Maximum seconds between XLSX flushes')
	p.add_argument('--dataset', default='embedding-data/PAQ_pairs', help='Source dataset name')
	p.add_argument('--workers', type=int, default=1,
				   help='Number of worker processes (master-slave mode when > 1)')
	p.add_argument('--batch-size', type=int, default=20,
				   help='Rows per batch sent to each worker (default: 20)')
	p.add_argument(
		'--temp-guard-max', type=int, default=80,
		help='Pause dispatching when GPU temp reaches this value (C). '
		'Only active when --device is cuda (default: 80)',
	)
	p.add_argument(
		'--temp-guard-resume', type=int, default=75,
		help='Resume dispatching once GPU temp drops to this value (C). '
		'Only active when --device is cuda (default: 75)',
	)
	p.add_argument(
		'--temp-guard-stop', type=int, default=90,
		help='Hard-kill the whole pipeline when GPU temp reaches this value (C). '
		'Only active when --device is cuda (default: 90)',
	)
	p.add_argument(
		'--temp-guard-time', type=int, default=30,
		help='Seconds between GPU temperature checks (default: 30)',
	)
	p.add_argument(
		'--temp-guard-gpu', type=int, default=0,
		help='GPU index to monitor with nvidia-smi (default: 0)',
	)
	p.add_argument(
		'--temp-guard-offload', action=argparse.BooleanOptionalAction, default=True,
		help='When pausing, move the model VRAM -> system RAM and reload it on '
		'resume (default: enabled). Use --no-temp-guard-offload to just idle '
		'the GPU instead of shuffling memory. Only used when --device is cuda.',
	)
	add_engine_args(p)
	return p


def main():
	parser = build_arg_parser()
	args = parser.parse_args()

	if args.workers <= 0:
		parser.error("--workers must be >= 1")
	if args.batch_size <= 0:
		parser.error("--batch-size must be >= 1")
	if args.nretries <= 0:
		parser.error("--nretries must be >= 1")
	if not (args.temp_guard_resume < args.temp_guard_max < args.temp_guard_stop):
		parser.error(
			"--temp-guard-resume must be < --temp-guard-max must be < --temp-guard-stop"
		)
	if args.temp_guard_time < 1:
		parser.error("--temp-guard-time must be >= 1")

	engine_config = engine_config_from_args(args)

	# Temperature guard is only available on CUDA devices.
	device_str = (args.device or "").lower()
	temp_guard = None
	if device_str.startswith("cuda"):
		temp_guard = TempGuard(
			temp_max=args.temp_guard_max,
			temp_resume=args.temp_guard_resume,
			temp_stop=args.temp_guard_stop,
			check_interval=args.temp_guard_time,
			gpu_index=args.temp_guard_gpu,
		)

	if args.workers == 1:
		translate_pairs_single(
			skip_rows=args.skip_rows,
			max_rows=args.max_rows,
			output_excel=args.output_excel,
			log_file=args.log_file,
			engine_config=engine_config,
			flush_every=args.flush_every,
			flush_interval_seconds=args.flush_interval_seconds,
			dataset_name=args.dataset,
			temp_guard=temp_guard,
			offload_enabled=temp_guard is not None and args.temp_guard_offload,
		)
	else:
		mp.freeze_support()
		coordinator = MasterCoordinator(
			output_excel=args.output_excel,
			log_file=args.log_file,
			engine_config=engine_config,
			num_workers=args.workers,
			batch_size=args.batch_size,
			skip_rows=args.skip_rows,
			max_rows=args.max_rows,
			flush_every=args.flush_every,
			flush_interval_seconds=args.flush_interval_seconds,
			dataset_name=args.dataset,
			temp_guard=temp_guard,
			offload_enabled=temp_guard is not None and args.temp_guard_offload,
		)
		coordinator.run()


if __name__ == '__main__':
	main()
