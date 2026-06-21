# src/inference/

Model inference with multi-backend support (HuggingFace, MLX, OpenAI, Anthropic).

## Quick Start

```python
from src.inference import ModelRunner

runner = ModelRunner("Qwen/Qwen3-0.6B")
traj = runner.generate_trajectory_from_prompt("Write a story", max_new_tokens=100)
```

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
  family's assistant-turn token and (reasoning-only) `<think>`/`</think>` markers
  so callers (e.g. SESGO geometry) can locate structural token positions per
  family — Qwen `<|im_start|>`, Llama `<|start_header_id|>`, Gemma
  `<start_of_turn>`, Mistral `[/INST]` — instead of hardcoding Qwen's tokens.
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

`structural_markers_for(name)` → `ChatTemplateMarkers` (turn marker + optional
`<think>`/`</think>`), resolved by instruct-family substring. Every marker is a
single token in its family's vocab, so position-finders can search the forced id
sequence directly. Surfaced on the runner via `ModelRunner.structural_markers`.

## Backends Directory

- `model_backend.py`: Base `Backend` abstract class
- `backend_huggingface.py`: HuggingFace + transformers
- `backend_mlx.py`: MLX for Apple Silicon
- `backend_openai.py`: OpenAI API
- `backend_anthropic.py`: Anthropic API (no logprobs)
- `backend_selection.py`: Hardware detection logic

See [EXPLANATION.md](./EXPLANATION.md) for detailed architecture and API specifications.
