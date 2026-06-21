"""Run SESGO prompts through a model at two levels and record the results.

Level 1 (non-thinking): teacher-force the three option labels, read a 3-way
softmax over the displayed POSITIONS, then remap that mass to the three MEANINGS
through the prompt's position_labels — invariant to ordering, which defeats
position bias. Level 2 (thinking): sample free-form generations, parse the role
each committed to, and normalize the counts into an empirical role distribution.
"""

from __future__ import annotations

import math

from src.common.device_utils import ProgressTracker, clear_gpu_memory
from src.common.logging import log
from src.datasets.prompt import SesgoPromptDataset, SesgoPromptSample
from src.ternary_choice import TernaryChoiceRunner
from .sesgo_dataset import SesgoDataset
from .sesgo_non_thinking import SesgoNonThinking
from .sesgo_query_config import SesgoQueryConfig
from .sesgo_response_parsing import parse_chosen_label
from .sesgo_sample import SesgoSample
from .sesgo_thinking import SesgoThinking, summarize_labels

# Periodically free accelerator memory so long runs don't accumulate caches.
_GPU_CLEAR_EVERY = 25


class SesgoQuerier:
    """Queries a TernaryChoiceRunner for both SESGO eval levels."""

    def __init__(self, config: SesgoQueryConfig):
        self.config = config
        self._runner: TernaryChoiceRunner | None = None

    def _load_model(self, name: str) -> TernaryChoiceRunner:
        """Construct (and cache) the runner; reuse it across samples/datasets."""
        if self._runner is not None and self._runner.model_name == name:
            return self._runner
        self._runner = TernaryChoiceRunner(model_name=name)
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
        nt = SesgoNonThinking.from_ternary(choice, sample.position_labels)

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
        thinking = completions = None
        if self.config.do_thinking:
            thinking, completions = self._thinking(prompt_sample, runner)

        return SesgoSample(
            sample_idx=prompt_sample.sample_idx,
            question_id=prompt_sample.question_id,
            scaffold_id=prompt_sample.scaffold_id,
            question_polarity=prompt_sample.question_polarity,
            bias_category=prompt_sample.bias_category,
            language=prompt_sample.language,
            label_style=prompt_sample.label_style,
            gold_label=prompt_sample.gold_label,
            prompt_text=prompt_sample.text,
            non_thinking=non_thinking,
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
        """Query every prompt in a dataset and collect a SesgoDataset."""
        runner = self._load_model(model_name)
        samples = self._subsample(prompt_dataset.samples)
        log(f"[sesgo] Querying {len(samples)} prompts with {runner.model_name}...")

        tracker = ProgressTracker(total=len(samples), progress_every=10, memory_every=50)
        results: list[SesgoSample] = []
        for i, prompt_sample in enumerate(samples):
            tracker.step(i)
            results.append(self.query_sample(prompt_sample, runner))
            if (i + 1) % _GPU_CLEAR_EVERY == 0:
                clear_gpu_memory()
        clear_gpu_memory()

        return SesgoDataset(
            prompt_dataset_id=prompt_dataset.dataset_id,
            model=model_name,
            config=self.config,
            samples=results,
        )
