# src/inference/

Model inference with multi-backend support (HuggingFace, MLX, OpenAI, Anthropic,
vLLM).

## Quick Start

```python
from src.inference import ModelRunner

runner = ModelRunner("Qwen/Qwen3-0.6B")
traj = runner.generate_trajectory_from_prompt("Write a story", max_new_tokens=100)

# Batched (many prompts in one forward pass; HF or vLLM backend)
texts = runner.generate_batch(["A", "B", "C"], max_new_tokens=64)
```

## Batched inference

For high throughput, the runner batches a chunk of prompts into a single forward
pass instead of looping one at a time:

- `generate_batch(prompts, ...)` — one batched decode (HF `model.generate` over a
  left-padded batch, or vLLM continuous batching).
- `compute_trajectories_batch(token_ids_batch)` — teacher-forced logprobs/logits
  for many sequences in one pass; left-pads + attention-masks so padding never
  leaks into attention, then slices each sample's real logits back out.
- `run_with_cache_batch(token_ids_batch, names_filter)` — activations for many
  sequences in one pass, each cache sliced back to its real length.

`batched_padding_helpers.py` is the single source of truth for **left padding**
(keeps the last real token at index `-1` and lets unpadded positions map by one
additive offset) + the attention mask. Verified on Qwen3-0.6B to match the
single-sample path within fp tolerance (probs ~1e-4, activations cos-sim
≥0.999996), and faster.

## ModelRunner

`ModelRunner` is the unified inference interface. It automatically detects and routes to the appropriate backend based on model name and hardware.

### Backend Selection

Priority order:
1. **OpenAI**: `openai/...`, `gpt-4`, `gpt-3`, `o1`, `o3` → OpenAI API
2. **Anthropic**: `anthropic/...`, `claude` → Anthropic API
3. **Gemini**: `gemini:...`, `gemini-...` → Gemini API. NOTE: the `google/` HF
   org prefix is **local** (e.g. `google/gemma-2-2b-it` loads HuggingFace
   weights, it is *not* the Gemini API).
4. **MLX**: Apple Silicon + MLX available → MLX (optimized)
5. **HuggingFace**: Default fallback

`detect_backend_for_name(name)` / `is_cloud_api_name(name)` (module-level in
`model_runner.py`) expose this routing so callers can decide cloud-vs-local
*before* constructing a runner — e.g. pipelines that pin the HuggingFace backend
for local models because MLX cannot load every instruct family.

### Model Loading

Models are loaded in `__init__` based on detected backend:

- **HuggingFace**: `AutoModelForCausalLM.from_pretrained()` with optional `torch.compile()` on CUDA
- **MLX**: `mlx_lm.load()` for Apple Silicon
- **OpenAI/Anthropic**: API clients initialized (no local model)

### Key Features

- **Auto chat model detection**: Detects instruct models by name patterns
- **Reasoning model detection**: Checks tokenizer's chat template for thinking tokens
- **Model-aware structural markers**: `runner.structural_markers` returns the
  family's assistant-turn token, the previous-turn closer (`turn_end`), the
  assistant role word (`assistant_role`), and (reasoning-only) `<think>`/
  `</think>` markers so callers (e.g. SESGO geometry) can locate fine-grained
  structural token positions per family — Qwen `<|im_start|>`/`<|im_end|>`,
  Llama `<|start_header_id|>`/`<|eot_id|>`, Gemma `<start_of_turn>`/
  `<end_of_turn>`, Mistral `[/INST]` — instead of hardcoding Qwen's tokens.
- **Encoding/decoding**: Unified tokenizer access regardless of backend
- **Trajectory generation**: Returns `GeneratedTrajectory` with logprobs

## GeneratedTrajectory

Extends `TokenTrajectory` with:
- `internals`: dict of captured activations from forward pass
- Methods: `from_inference()`, `from_logprobs()`, `from_token_trajectory()`

## EmbeddingRunner

Uses sentence-transformers for text embeddings and similarity scoring.

```python
from src.inference import EmbeddingRunner

runner = EmbeddingRunner()
sim = runner.similarity("hello", "hi")
sims = runner.similarities("hello", ["hi", "bye"])
```

## chat_template_markers.py

`structural_markers_for(name)` → `ChatTemplateMarkers` (turn marker + previous-
turn closer `turn_end` + role word `assistant_role` + optional `<think>`/
`</think>`), resolved by instruct-family substring. The single-token markers are
located by exact id; the multi-token role word is matched by its first token
after the turn opener, so position-finders can search the forced id sequence
directly. Surfaced on the runner via `ModelRunner.structural_markers`.

## Backends Directory

- `model_backend.py`: Base `Backend` abstract class. `forward` / `run_with_cache`
  take an optional `attention_mask` (required for a padded multi-sample batch;
  `None` is the single-sample fast path).
- `backend_huggingface.py`: HuggingFace + transformers. Honors the attention mask;
  adds `generate_batch` (one padded `model.generate`).
- `backend_mlx.py`: MLX for Apple Silicon
- `backend_openai.py`: OpenAI API
- `backend_anthropic.py`: Anthropic API (no logprobs)
- `backend_selection.py`: Hardware detection logic

## vLLM backend (CUDA-only, cloud fast path)

`vllm_batched_backend.py` (`ModelBackend.VLLM`) is the high-throughput cloud
backend: continuous-batching `generate_batch` plus teacher-forced
`score_options_batch` (option-token logprobs via `prompt_logprobs`, in
`vllm_option_scoring.py`). Activations / interventions / raw-logit `forward` are
unsupported by design and raise a clear message routing to HuggingFace (the SESGO
geometry driver already forces HF). vLLM is **CUDA-only**: the import is guarded
and construction raises an actionable error off a GPU box, so the module imports
fine on Apple Silicon. Installed via the `cloud` extra (`pip install .[cloud]`),
which is a no-op on Darwin.

See [EXPLANATION.md](./EXPLANATION.md) for detailed architecture and API specifications.
