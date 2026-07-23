"""GPU temperature guard.

Polls the GPU temperature via the ``nvidia-smi`` command and exposes a
background monitor that can pause/resume translation work and, in an
emergency, request a hard kill of the whole pipeline.

Usage::

	from gpu_temp_guard import TempGuard
	tg = TempGuard(temp_max=80, temp_resume=75, temp_stop=90, check_interval=30)
	tg.start()
	if tg.pause_event.is_set():
		# don't dispatch new work
	if tg.kill_event.is_set():
		# terminate everything
	tg.stop()

The guard is self-contained: ``read_gpu_temperature`` degrades gracefully
(returning ``None``) whenever ``nvidia-smi`` is missing, fails, or yields
unparseable output, so the monitor simply skips that cycle.
"""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# nvidia-smi parsing
# ---------------------------------------------------------------------------
def read_gpu_temperature(gpu_index: int = 0) -> Optional[int]:
	"""Return the temperature (Celsius) of GPU *gpu_index*, or ``None``.

	Uses the query form of ``nvidia-smi`` which prints one temperature per
	line (just the integer, no units). Output example for two GPUs::

		72
		68
	"""
	try:
		result = subprocess.run(
			[
				"nvidia-smi",
				"--query-gpu=temperature.gpu",
				"--format=csv,noheader,nounits",
			],
			capture_output=True,
			text=True,
			timeout=10,
		)
	except (FileNotFoundError, subprocess.TimeoutExpired):
		return None

	if result.returncode != 0:
		return None

	lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
	if gpu_index < 0 or gpu_index >= len(lines):
		return None
	try:
		return int(lines[gpu_index])
	except ValueError:
		return None


# ---------------------------------------------------------------------------
# Temperature guard
# ---------------------------------------------------------------------------
class TempGuard:
	"""Daemon thread that watches the GPU temperature and signals the
	pipeline to pause/resume/kill.

	* ``pause_event`` is set while ``temp_max <= temp`` and only cleared
	  once the reading drops back to ``temp_resume`` or below (hysteresis
	  so the workload does not flap around the threshold).
	* ``kill_event`` is latched when ``temp >= temp_stop``; the caller is
	  expected to terminate the pipeline immediately.
	"""

	def __init__(
		self,
		temp_max: int,
		temp_resume: int,
		temp_stop: int,
		check_interval: int,
		gpu_index: int = 0,
		log_fn: Callable[[str], None] = print,
	):
		if not (temp_resume < temp_max < temp_stop):
			raise ValueError(
				f"Invalid thresholds: need resume({temp_resume}) < "
				f"max({temp_max}) < stop({temp_stop})"
			)
		if check_interval < 1:
			raise ValueError("check_interval must be >= 1 second")

		self.temp_max = temp_max
		self.temp_resume = temp_resume
		self.temp_stop = temp_stop
		self.check_interval = check_interval
		self.gpu_index = gpu_index
		self._log = log_fn

		# Shared state
		self.pause_event = threading.Event()
		self.kill_event = threading.Event()
		# offload_event mirrors pause_event: set when the GPU model should be
		# moved VRAM -> system RAM, cleared when it can be reloaded to GPU.
		self.offload_event = threading.Event()

		# Internal stop signal for the monitor thread
		self._stop_thread = threading.Event()
		self._thread: Optional[threading.Thread] = None

	def start(self) -> None:
		"""Spawn the background monitor thread (idempotent)."""
		if self._thread is not None and self._thread.is_alive():
			return
		self._thread = threading.Thread(
			target=self._run, name="TempGuard", daemon=True
		)
		self._thread.start()

	def stop(self) -> None:
		"""Signal the monitor thread to exit and wait briefly."""
		self._stop_thread.set()
		if self._thread is not None:
			self._thread.join(timeout=self.check_interval + 1)
			self._thread = None

	def _run(self) -> None:
		while not self._stop_thread.is_set():
			temp = read_gpu_temperature(self.gpu_index)
			if temp is None:
				# nvidia-smi unavailable / failed: skip cycle, keep going.
				self._stop_thread.wait(self.check_interval)
				continue

			# Critical threshold: latch kill and exit the thread.
			if temp >= self.temp_stop:
				self.kill_event.set()
				self.pause_event.set()
				self.offload_event.set()
				self._log(
					f"[TempGuard] STOP: {temp}C >= {self.temp_stop}C "
					f"(GPU {self.gpu_index}) - killing pipeline"
				)
				return

			# Pause when crossing max upward.
			if temp >= self.temp_max:
				if not self.pause_event.is_set():
					self.pause_event.set()
					self.offload_event.set()
					self._log(
						f"[TempGuard] PAUSE: {temp}C >= {self.temp_max}C "
						f"(GPU {self.gpu_index})"
					)
			# Resume only when back below the resume threshold (hysteresis).
			elif temp <= self.temp_resume:
				if self.pause_event.is_set():
					self.pause_event.clear()
					self.offload_event.clear()
					self._log(
						f"[TempGuard] RESUME: {temp}C <= {self.temp_resume}C "
						f"(GPU {self.gpu_index})"
					)

			# Interruptible sleep.
			self._stop_thread.wait(self.check_interval)


# ---------------------------------------------------------------------------
# Smoke test: python gpu_temp_guard.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
	def _now() -> str:
		return time.strftime("%H:%M:%S")

	tg = TempGuard(
		temp_max=80,
		temp_resume=75,
		temp_stop=90,
		check_interval=5,
		log_fn=lambda m: print(f"{_now()} {m}"),
	)
	print(f"Starting TempGuard (gpu={tg.gpu_index}). Ctrl-C to stop.")
	tg.start()
	try:
		while not tg.kill_event.is_set():
			t = read_gpu_temperature(tg.gpu_index)
			state = "PAUSED" if tg.pause_event.is_set() else "running"
			print(f"{_now()} temp={t}C state={state}")
			time.sleep(5)
	except KeyboardInterrupt:
		pass
	finally:
		tg.stop()
		print("TempGuard stopped.")
