"""vLLM CUDA backend: continuous-batching generation + teacher-forced scoring.

vLLM is the high-throughput fast path for the SESGO pipeline on a cloud GPU box.
Its continuous batching keeps the GPU saturated across many in-flight sequences,
so the generation-heavy studies (baseline greedy decode, divergence/selection
thinking draws) collapse into ONE batched call instead of a per-sample loop.

It also scores the teacher-forced choose3 option tokens in a single batched pass
via ``prompt_logprobs`` over the three forced continuations.

Scope (deliberate): vLLM does not expose residual activations or accept
intervention hooks, so geometry / interventions stay on the HuggingFace backend
(the SESGO geometry driver already forces HF). Those methods raise a clear error.

vLLM is CUDA-only and is NOT installable on Apple Silicon: the import is guarded
and construction raises with an actionable message off a CUDA box. The batching
LOGIC is verified locally through the HuggingFace batched path; this backend is
the cloud-image equivalent.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from .backends.model_backend import Backend
from .interventions import Intervention
from .vllm_option_scoring import option_token_logprobs

_NO_INTERNALS = (
    "vLLM exposes no residual activations or intervention hooks. Use the "
    "HuggingFace backend for geometry / interventions (the geometry driver "
    "already forces it)."
)


def _require_vllm():
    """Import vLLM or raise an actionable, CUDA-only error message."""
    try:
        from vllm import LLM, SamplingParams  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only on a GPU box
        raise RuntimeError(
            "vLLM is not installed. It is CUDA-only and cannot run on Apple "
            "Silicon. Install it on the cloud GPU image (pip install vllm) or "
            "use the HuggingFace backend locally."
        ) from exc
    if not torch.cuda.is_available():  # pragma: no cover - GPU-only
        raise RuntimeError(
            "vLLM requires a CUDA GPU; none is available. Select the HuggingFace "
            "backend on non-CUDA hosts."
        )
    return LLM, SamplingParams


class VLLMBackend(Backend):
    """High-throughput batched backend backed by a vLLM engine (CUDA only)."""

    # vLLM manages its own CUDA graphs / memory; torch.inference_mode around its
    # engine calls is unnecessary and can conflict, so opt out.
    supports_inference_mode = False

    def __init__(self, runner: Any, model_name: str, dtype: str = "float16"):
        super().__init__(runner)
        LLM, SamplingParams = _require_vllm()
        self._SamplingParams = SamplingParams
        # One engine per model; vLLM handles paged-attention + continuous batching.
        self._llm = LLM(model=model_name, dtype=dtype, trust_remote_code=True)
        self._tokenizer = self._llm.get_tokenizer()
        cfg = self._llm.llm_engine.model_config
        self._n_layers = cfg.get_num_layers(self._llm.llm_engine.parallel_config)
        self._d_model = cfg.get_hidden_size()

    # ---- tokenizer / shape -------------------------------------------------

    def get_tokenizer(self):
        return self._tokenizer

    def get_n_layers(self) -> int:
        return self._n_layers

    def get_d_model(self) -> int:
        return self._d_model

    def encode(
        self, text: str, add_special_tokens: bool = True, prepend_bos: bool = False
    ) -> torch.Tensor:
        ids = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return torch.tensor([ids])

    def decode(self, token_ids: torch.Tensor | list) -> str:
        ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    # ---- generation (vLLM's core strength) ---------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        intervention: Optional[Intervention] = None,
        past_kv_cache: Any = None,
    ) -> str:
        """Single-prompt generation (delegates to the batched path of size 1)."""
        if intervention is not None:
            raise NotImplementedError(_NO_INTERNALS)
        return self.generate_batch([prompt], max_new_tokens, temperature)[0]

    def generate_batch(
        self,
        prompts: Sequence[str],
        max_new_tokens: int,
        temperature: float,
    ) -> list[str]:
        """Decode many prompts at once via vLLM continuous batching.

        ``prompts`` are already chat-templated + prefilled by ModelRunner, so we
        pass them verbatim (no extra special tokens). Returns each continuation.
        """
        if not prompts:
            return []
        params = self._SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_new_tokens,
            # 0 temperature -> greedy; vLLM treats temperature==0 as argmax.
        )
        outputs = self._llm.generate(list(prompts), params, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    # ---- teacher-forced option scoring (batched) ---------------------------

    def score_options_batch(
        self,
        forced_texts: list[str],
        option_token_ids: list[int],
    ) -> list[float]:
        """Conditional logprob of each forced text's final option token.

        ``forced_texts[i]`` is prompt+prefix+option_i fully rendered; we request
        prompt_logprobs and read the logprob of ``option_token_ids[i]`` at the
        position it occupies. One batched vLLM call scores every option.
        """
        return option_token_logprobs(
            self._llm, self._SamplingParams, forced_texts, option_token_ids
        )

    def get_next_token_probs(
        self, prompt: str, target_tokens: Sequence[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        ids = [self._tokenizer.encode(t, add_special_tokens=False)[0] for t in target_tokens]
        forced = [prompt + t for t in target_tokens]
        lps = self.score_options_batch(forced, ids)
        return {t: float(torch.tensor(lp).exp()) for t, lp in zip(target_tokens, lps)}

    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: Sequence[int], past_kv_cache: Any = None
    ) -> dict[int, float]:
        forced = [prompt + self._tokenizer.decode([tid]) for tid in token_ids]
        lps = self.score_options_batch(forced, list(token_ids))
        return {tid: float(torch.tensor(lp).exp()) for tid, lp in zip(token_ids, lps)}

    # ---- unsupported: activations / interventions / raw logits -------------

    def forward(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        raise NotImplementedError(
            "vLLM does not return full-vocab logits tensors. Use score_options_batch "
            "for teacher-forced scoring, or the HuggingFace backend for raw logits."
        )

    def run_with_cache(self, input_ids, names_filter, past_kv_cache=None, attention_mask=None):
        raise NotImplementedError(_NO_INTERNALS)

    def run_with_cache_and_grad(self, input_ids, names_filter):
        raise NotImplementedError(_NO_INTERNALS)

    def run_with_intervention(self, input_ids, interventions):
        raise NotImplementedError(_NO_INTERNALS)

    def run_with_intervention_and_cache(self, input_ids, interventions, names_filter):
        raise NotImplementedError(_NO_INTERNALS)

    def generate_trajectory(self, token_ids, max_new_tokens, temperature):
        raise NotImplementedError(
            "Use generate_batch (vLLM) for generation; trajectory logprobs come "
            "from score_options_batch for teacher-forced paths."
        )

    def generate_from_cache(self, prefill_logits, frozen_kv_cache, max_new_tokens, temperature):
        raise NotImplementedError("vLLM manages its own KV cache internally.")

    def init_kv_cache(self):
        return None
