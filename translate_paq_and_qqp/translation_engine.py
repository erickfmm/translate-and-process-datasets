"""Translation engine abstraction with retry/timeout support.

Two backends are exposed through the single :func:`make_engine` factory:

* ``transformers`` - HuggingFace ``AutoModelForSeq2SeqLM`` (e.g.
  ``Helsinki-NLP/opus-mt-en-es``). Runs in-process on CPU/CUDA.
* ``ollama`` - a local Ollama server (e.g. with ``translategemma:latest``
  pulled). One HTTP request is issued per text.

``torch`` / ``transformers`` are imported lazily inside
``TransformersEngine.__init__`` so that Ollama-only users do not need them
installed (and therefore do not need to download the large CUDA wheels).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Ollama prompt
# ---------------------------------------------------------------------------
# Two literal blank lines between the instruction and the source text, exactly
# as specified for the local translategemma:latest model card.
_OLLAMA_PROMPT_HEADER = (
	"You are a professional English (en) to Spanish (es) translator. "
	"Your goal is to accurately convey the meaning and nuances of the original "
	"English text while adhering to Spanish grammar, vocabulary, and cultural "
	"sensitivities.\n"
	"Produce only the Spanish translation, without any additional explanations "
	"or commentary. Please translate the following English text into Spanish:\n"
	"\n"
	"\n"
)


def build_ollama_prompt(text: str) -> str:
	"""Wrap *text* with the translategemma instruction header."""
	return _OLLAMA_PROMPT_HEADER + text


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class RetryExhaustedError(RuntimeError):
	"""Raised after every retry attempt has failed.

	Callers can catch this specifically to decide whether to abort the whole
	pipeline (as opposed to skipping just the offending item).
	"""

	def __init__(self, attempts: int, last_exc: Optional[BaseException]):
		super().__init__(
			f"All {attempts} retry attempt(s) failed. Last error: {last_exc!r}"
		)
		self.attempts = attempts
		self.last_exc = last_exc


# ---------------------------------------------------------------------------
# Engine interface
# ---------------------------------------------------------------------------
class TranslationEngine:
	"""Maps a list of source strings to a list of translated strings."""

	name: str = "base"

	def translate(self, texts: List[str]) -> List[str]:  # pragma: no cover - abstract
		raise NotImplementedError

	def offload_to_cpu(self) -> None:
		"""Move model weights to system RAM to free GPU VRAM.

		Default no-op; overridden by engines that own an in-process model.
		"""
		return None

	def reload_to_device(self) -> None:
		"""Move model weights back to the inference device.

		Default no-op; overridden by engines that own an in-process model.
		"""
		return None


class _RetryMixin:
	"""Mixin providing ``_call_with_retry`` with exponential backoff."""

	nretries: int = 1
	_retry_base_delay: float = 1.0

	def _call_with_retry(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
		last_exc: Optional[BaseException] = None
		for attempt in range(self.nretries):
			try:
				return fn(*args, **kwargs)
			except Exception as exc:  # noqa: BLE001 - intentional broad catch
				last_exc = exc
				remaining = self.nretries - attempt - 1
				if remaining > 0:
					delay = self._retry_base_delay * (2 ** attempt)
					print(
						f"[engine:{type(self).__name__}] attempt {attempt + 1}/{self.nretries} "
						f"failed: {exc!r}; retrying in {delay:.1f}s ({remaining} left)"
					)
					time.sleep(delay)
				else:
					print(
						f"[engine:{type(self).__name__}] attempt {attempt + 1}/{self.nretries} "
						f"failed: {exc!r}; giving up"
					)
		raise RetryExhaustedError(self.nretries, last_exc)


# ---------------------------------------------------------------------------
# Transformers backend
# ---------------------------------------------------------------------------
class TransformersEngine(_RetryMixin, TranslationEngine):
	"""HuggingFace Seq2Seq engine (opus-mt style models)."""

	name = "transformers"

	def __init__(
		self,
		model_name: str = "Helsinki-NLP/opus-mt-en-es",
		device: Optional[str] = None,
		timeout: float = 300.0,
		nretries: int = 3,
		**_unused: Any,
	):
		# Lazy imports: keep torch/transformers out of the import graph for
		# Ollama-only users.
		import torch
		from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

		self.torch = torch
		self._model_name = model_name
		# timeout is informational for the in-process engine; generation calls
		# are CPU/GPU-bound and not network-bound.
		self.timeout = timeout
		self.nretries = max(1, int(nretries))
		self.device = self._resolve_device(device, torch)

		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
		self.model.to(self.device)
		self.model.eval()

	@staticmethod
	def _resolve_device(device: Optional[str], torch) -> str:
		if device is None:
			return "cuda" if torch.cuda.is_available() else "cpu"
		try:
			resolved = str(torch.device(device))
		except (TypeError, RuntimeError) as exc:
			raise ValueError(
				f"Invalid device '{device}'. Use values like 'cpu', 'cuda', or 'cuda:0'."
			) from exc
		if resolved.startswith("cuda") and not torch.cuda.is_available():
			raise RuntimeError(
				"CUDA device requested, but CUDA is not available in this environment."
			)
		return resolved

	def translate(self, texts: List[str]) -> List[str]:
		tokenizer = self.tokenizer
		model = self.model
		device = self.device

		def _do() -> List[str]:
			encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
			encoded = {k: v.to(device) for k, v in encoded.items()}
			tokens = model.generate(**encoded)
			return tokenizer.batch_decode(tokens, skip_special_tokens=True)

		with self.torch.inference_mode():
			return self._call_with_retry(_do)

	def offload_to_cpu(self) -> None:
		"""Move the model VRAM -> system RAM and release the CUDA cache so
		nvidia-smi reports the freed memory (lets the GPU cool down)."""
		self.model.to("cpu")
		if self.torch.cuda.is_available():
			self.torch.cuda.empty_cache()

	def reload_to_device(self) -> None:
		"""Move the model back to the configured inference device (GPU)."""
		self.model.to(self.device)


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------
class OllamaEngine(_RetryMixin, TranslationEngine):
	"""Ollama HTTP engine. One ``generate`` request per text."""

	name = "ollama"

	def __init__(
		self,
		model: str = "translategemma:latest",
		host: str = "http://localhost:11434",
		timeout: float = 300.0,
		nretries: int = 3,
		**_unused: Any,
	):
		import ollama  # lazy

		self.nretries = max(1, int(nretries))
		self.timeout = timeout
		self.model = model
		self.host = host
		self.client = ollama.Client(host=host, timeout=timeout)

	def _generate_one(self, text: str) -> str:
		prompt = build_ollama_prompt(text)

		def _do() -> str:
			resp = self.client.generate(
				model=self.model,
				prompt=prompt,
				stream=False,
				options={"temperature": 0},
			)
			# Ollama's response object supports both dict-style and attribute access.
			if isinstance(resp, dict):
				content = resp.get("response", "")
			else:
				content = getattr(resp, "response", "") or ""
			return content.strip()

		return self._call_with_retry(_do)

	def translate(self, texts: List[str]) -> List[str]:
		return [self._generate_one(t) for t in texts]


# ---------------------------------------------------------------------------
# Factory + argparse helper
# ---------------------------------------------------------------------------
def make_engine(
	engine: str,
	*,
	model_name: str = "Helsinki-NLP/opus-mt-en-es",
	device: Optional[str] = None,
	ollama_model: str = "translategemma:latest",
	ollama_host: str = "http://localhost:11434",
	timeout: float = 300.0,
	nretries: int = 3,
) -> TranslationEngine:
	"""Construct a :class:`TranslationEngine` by name."""
	if engine == "transformers":
		return TransformersEngine(
			model_name=model_name,
			device=device,
			timeout=timeout,
			nretries=nretries,
		)
	if engine == "ollama":
		return OllamaEngine(
			model=ollama_model,
			host=ollama_host,
			timeout=timeout,
			nretries=nretries,
		)
	raise ValueError(
		f"Unknown engine '{engine}'. Use 'transformers' or 'ollama'."
	)


def engine_config_from_args(args) -> Dict[str, Any]:
	"""Build the kwargs dict for :func:`make_engine` from an argparse Namespace."""
	return {
		"engine": args.engine,
		"model_name": args.model,
		"device": args.device,
		"ollama_model": args.ollama_model,
		"ollama_host": args.ollama_host,
		"timeout": args.timeout,
		"nretries": args.nretries,
	}


def add_engine_args(parser) -> None:
	"""Register the engine / retry / ollama CLI flags on *parser*.

	Call this alongside the script-specific arguments. The ``--model`` and
	``--device`` flags are also added here so behaviour is consistent across
	scripts (``--model``/``--device`` only apply when ``--engine transformers``).
	"""
	parser.add_argument(
		"--engine",
		choices=["transformers", "ollama"],
		default="transformers",
		help="Translation backend (default: transformers)",
	)
	parser.add_argument(
		"--model",
		default="Helsinki-NLP/opus-mt-en-es",
		help="HuggingFace model name (only used when --engine transformers)",
	)
	parser.add_argument(
		"--device",
		default=None,
		help="Device for inference, e.g. cpu, cuda, cuda:0 (only used when --engine transformers)",
	)
	parser.add_argument(
		"--ollama-model",
		default="translategemma:latest",
		help="Ollama model tag (only used when --engine ollama)",
	)
	parser.add_argument(
		"--ollama-host",
		default="http://localhost:11434",
		help="Ollama server URL (only used when --engine ollama)",
	)
	parser.add_argument(
		"--timeout",
		type=float,
		default=300.0,
		help="Per-request timeout in seconds (applies to --engine ollama; "
		"informational for transformers). Default: 300 (5 minutes)",
	)
	parser.add_argument(
		"--nretries",
		type=int,
		default=3,
		help="Number of attempts per translation before giving up. "
		"Exponential backoff between attempts. Default: 3",
	)
