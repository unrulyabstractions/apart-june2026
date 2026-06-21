"""Run SESGO prompts through a model at two levels and record the results.

Level 1 (non-thinking): teacher-force the three option labels, read a 3-way
softmax over the displayed POSITIONS, then remap that mass to the three MEANINGS
through the prompt's position_labels — invariant to ordering, which defeats
position bias. Level 2 (thinking): sample free-form generations, parse the role
each committed to, and normalize the counts into an empirical role distribution.
"""

from __future__ import annotations

import math

from src.binary_choice import BinaryChoiceRunner
from src.common.device_utils import ProgressTracker, clear_gpu_memory
from src.common.logging import log
from src.datasets.prompt import SesgoPromptDataset, SesgoPromptSample
from src.inference.backends import ModelBackend
from src.inference.model_runner import is_cloud_api_name
from src.ternary_choice import TernaryChoiceRunner
from .sesgo_batched_query import query_chunk
from .sesgo_dataset import SesgoDataset
from .sesgo_greedy_thinking import SesgoGreedyThinking
from .sesgo_non_thinking import SesgoNonThinking
from .sesgo_query_config import SesgoQueryConfig
from .sesgo_response_parsing import parse_chosen_label
from .sesgo_sample import SesgoSample
from .sesgo_thinking import SesgoThinking, summarize_labels
from .sesgo_two_option import SesgoTwoOption

# Periodically free accelerator memory so long runs don't accumulate caches.
_GPU_CLEAR_EVERY = 25


class SesgoQuerier:
    """Queries a TernaryChoiceRunner for both SESGO eval levels."""

    def __init__(self, config: SesgoQueryConfig):
        self.config = config
        self._runner: TernaryChoiceRunner | None = None
        self._binary: BinaryChoiceRunner | None = None

    def _binary_runner(self, runner: TernaryChoiceRunner) -> BinaryChoiceRunner:
        """A BinaryChoiceRunner over the SAME loaded weights (no second load)."""
        if self._binary is None or self._binary.model_name != runner.model_name:
            self._binary = BinaryChoiceRunner.from_runner(runner)
        return self._binary

    def _load_model(self, name: str) -> TernaryChoiceRunner:
        """Construct (and cache) the runner; reuse it across samples/datasets.

        Forces the HuggingFace backend for these local SESGO studies. The
        Apple-Silicon default (MLX) cannot reliably load the non-Qwen instruct
        families this pipeline now targets (Llama/Gemma/Mistral), so HF — which
        loads any HF causal-LM on CPU/MPS/CUDA — is the robust cross-model path.
        Cloud-API names (claude/gpt/gemini) auto-detect their backend, so we only
        pin HF when the name resolves to a local model. Geometry builds its own HF
        runner and injects it as `_runner`, bypassing this path entirely.
        """
        if self._runner is not None and self._runner.model_name == name:
            return self._runner
        backend = None if is_cloud_api_name(name) else ModelBackend.HUGGINGFACE
        self._runner = TernaryChoiceRunner(model_name=name, backend=backend)
        return self._runner

    def _non_thinking(
        self, sample: SesgoPromptSample, runner: TernaryChoiceRunner
    ) -> SesgoNonThinking:
        """Teacher-forced 3-way readout PLUS a greedy non-thinking decode.

        choose3 gives the per-option-path scores (remapped to roles); the greedy
        decode (skip-thinking prefill, temperature 0) gives the role the model
        actually emits when answering without reasoning.
        """
        prefix = sample.choice_prefix or "Answer: "
        choice = runner.choose3(sample.text, prefix, sample.option_labels)
        # from_ternary scatters each position's scores into its canonical role
        # slot via position_labels — invariant to which slot the role occupied.
        # These choose3 per-option metrics (and the argmax `predicted`) are the
        # cheap teacher-forced readout, so they are ALWAYS computed.
        nt = SesgoNonThinking.from_ternary(choice, sample.position_labels)

        # The greedy decode is a SECOND generation (skip-thinking prefill, temp 0).
        # Skip it for cheap label-only runs; nt then keeps its greedy defaults
        # (greedy_text="", greedy_label=None, decoding_mismatch=False).
        if self.config.do_greedy:
            greedy = runner.generate(
                sample.text,
                max_new_tokens=24,
                temperature=0.0,
                prefilling=runner.skip_thinking_prefix + prefix,
            )
            nt.greedy_text = greedy.strip()[:200]
            nt.greedy_label = parse_chosen_label(greedy, sample)
            nt.decoding_mismatch = (
                nt.greedy_label is not None and nt.greedy_label != nt.predicted
            )
        return nt

    def _non_thinking_2opt(
        self, sample: SesgoPromptSample, runner: TernaryChoiceRunner
    ) -> SesgoTwoOption:
        """Teacher-forced 2-way forced-choice readout (target vs other, no UNKNOWN).

        Runs choose2 over the 2-option prompt's two group markers, then remaps the
        binary preference through position_labels_2opt into [OTHER, TARGET] roles.
        Reuses the loaded weights via a wrapping BinaryChoiceRunner.
        """
        prefix = sample.choice_prefix or "Answer: "
        binary = self._binary_runner(runner)
        choice = binary.choose2(sample.text_2opt, prefix, sample.option_labels_2opt)
        return SesgoTwoOption.from_binary(choice, sample.position_labels_2opt)

    def _greedy_thinking(
        self, sample: SesgoPromptSample, runner: TernaryChoiceRunner
    ) -> SesgoGreedyThinking:
        """One DETERMINISTIC decode WITH reasoning, parsed for the committed role.

        Unlike the greedy non-thinking decode (skip-thinking prefill), this passes
        NO prefilling, so a reasoning model thinks before answering; we then parse
        the post-</think> answer via the shared parse_chosen_label. Temperature 0
        makes it the single answer the model commits to when it reasons greedily.
        """
        generated = runner.generate(
            sample.text,
            max_new_tokens=self.config.max_new_tokens,
            temperature=0.0,
        )
        return SesgoGreedyThinking(
            label=parse_chosen_label(generated, sample),
            text=generated.strip()[:200],
        )

    def _thinking(
        self, sample: SesgoPromptSample, runner: TernaryChoiceRunner
    ) -> tuple[SesgoThinking, list[str]]:
        """Sample N free-form draws, parse each role, summarize mean/std."""
        completions = [
            runner.generate(
                sample.text,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
            for _ in range(self.config.n_thinking_samples)
        ]
        labels = [
            label
            for label in (parse_chosen_label(c, sample) for c in completions)
            if label is not None
        ]
        return summarize_labels(labels), completions

    def query_sample(
        self, prompt_sample: SesgoPromptSample, runner: TernaryChoiceRunner
    ) -> SesgoSample:
        """Run both enabled levels for one prompt and assemble the record."""
        non_thinking = (
            self._non_thinking(prompt_sample, runner)
            if self.config.do_non_thinking
            else None
        )
        non_thinking_2opt = (
            self._non_thinking_2opt(prompt_sample, runner)
            if self.config.do_two_option
            else None
        )
        greedy_thinking = (
            self._greedy_thinking(prompt_sample, runner)
            if self.config.do_greedy_thinking
            else None
        )
        thinking = completions = None
        if self.config.do_thinking:
            thinking, completions = self._thinking(prompt_sample, runner)

        return SesgoSample(
            sample_idx=prompt_sample.sample_idx,
            question_id=prompt_sample.question_id,
            scaffold_id=prompt_sample.scaffold_id,
            question_polarity=prompt_sample.question_polarity,
            bias_category=prompt_sample.bias_category,
            context_condition=prompt_sample.context_condition,
            language=prompt_sample.language,
            label_style=prompt_sample.label_style,
            gold_label=prompt_sample.gold_label,
            prompt_text=prompt_sample.text,
            bbq=prompt_sample.bbq,
            target_identity=prompt_sample.target_identity,
            other_identity=prompt_sample.other_identity,
            non_thinking=non_thinking,
            non_thinking_2opt=non_thinking_2opt,
            greedy_thinking=greedy_thinking,
            thinking=thinking,
            _thinking_completions=completions,
        )

    def _subsample(self, samples: list[SesgoPromptSample]) -> list[SesgoPromptSample]:
        """Deterministically keep an EVENLY-SPACED ``subsample`` fraction.

        Scaffold conditions are emitted as contiguous blocks per item, so a
        leading slice would only ever cover the first condition (defeating the
        with/without-scaffold comparison on a partial run). An evenly-spaced
        stride spans all conditions/permutations/categories while staying
        reproducible without a seed.
        """
        frac = self.config.subsample
        if frac >= 1.0 or not samples:
            return samples
        n = max(1, math.ceil(len(samples) * frac))
        stride = max(1, len(samples) // n)
        return samples[::stride][:n]

    def query_dataset(
        self, prompt_dataset: SesgoPromptDataset, model_name: str
    ) -> SesgoDataset:
        """Query every prompt in a dataset and collect a SesgoDataset.

        With ``config.batch_size > 1`` the prompts are processed in chunks, each
        chunk's choose3 / greedy / thinking draws collapsed into batched forward
        passes (``sesgo_batched_query.query_chunk``). ``batch_size == 1`` keeps the
        exact single-sample path.
        """
        runner = self._load_model(model_name)
        samples = self._subsample(prompt_dataset.samples)
        bs = max(1, self.config.batch_size)
        log(
            f"[sesgo] Querying {len(samples)} prompts with {runner.model_name} "
            f"(batch_size={bs})..."
        )

        results = self._collect_samples(samples, runner, bs)

        return SesgoDataset(
            prompt_dataset_id=prompt_dataset.dataset_id,
            model=model_name,
            config=self.config,
            samples=results,
        )

    def _collect_samples(
        self, samples: list[SesgoPromptSample], runner: TernaryChoiceRunner, bs: int
    ) -> list[SesgoSample]:
        """Iterate the prompts in chunks of ``bs``, freeing memory periodically."""
        tracker = ProgressTracker(total=len(samples), progress_every=10, memory_every=50)
        results: list[SesgoSample] = []
        for start in range(0, len(samples), bs):
            tracker.step(start)
            chunk = samples[start : start + bs]
            if bs == 1:
                results.append(self.query_sample(chunk[0], runner))
            else:
                results.extend(query_chunk(chunk, runner, self.config))
            if (start // bs + 1) % _GPU_CLEAR_EVERY == 0:
                clear_gpu_memory()
        clear_gpu_memory()
        return results
