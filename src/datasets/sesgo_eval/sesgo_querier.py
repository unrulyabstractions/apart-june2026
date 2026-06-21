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
from src.datasets.sesgo import SesgoLabel
from src.ternary_choice import TernaryChoiceRunner
from .sesgo_dataset import SesgoDataset
from .sesgo_label_distribution import SesgoLabelDistribution
from .sesgo_query_config import SesgoQueryConfig
from .sesgo_response_parsing import parse_chosen_label
from .sesgo_sample import SesgoSample

# Periodically free accelerator memory so long runs don't accumulate caches.
_GPU_CLEAR_EVERY = 25
# Canonical role order shared by the distribution's (p_target, p_other, p_unknown).
_ROLE_ORDER = (SesgoLabel.TARGET, SesgoLabel.OTHER, SesgoLabel.UNKNOWN)


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
    ) -> SesgoLabelDistribution:
        """Teacher-forced 3-way softmax remapped from positions to roles."""
        choice = runner.choose3(
            sample.text, sample.choice_prefix or "Answer: ", sample.option_labels
        )
        # Move each position's probability into its role bucket — invariant to
        # which slot the role occupied.
        buckets = {role: 0.0 for role in _ROLE_ORDER}
        for i, prob in enumerate(choice.probs):
            buckets[sample.position_labels[i]] += prob
        return SesgoLabelDistribution.from_label_probs(
            buckets[SesgoLabel.TARGET],
            buckets[SesgoLabel.OTHER],
            buckets[SesgoLabel.UNKNOWN],
            n=1,
        )

    def _thinking(
        self, sample: SesgoPromptSample, runner: TernaryChoiceRunner
    ) -> tuple[SesgoLabelDistribution, list[str]]:
        """Sample N free-form draws, parse each role, count into a distribution."""
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
        counts = {role: labels.count(role) for role in _ROLE_ORDER}
        dist = SesgoLabelDistribution.from_counts(
            counts[SesgoLabel.TARGET],
            counts[SesgoLabel.OTHER],
            counts[SesgoLabel.UNKNOWN],
            n=len(labels),
        )
        return dist, completions

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
        """Deterministically keep the first ``subsample`` fraction of prompts.

        A leading slice (not a random sample) keeps runs reproducible without a
        seed and preserves the per-item prompt grouping in dataset order.
        """
        frac = self.config.subsample
        if frac >= 1.0 or not samples:
            return samples
        n = max(1, math.ceil(len(samples) * frac))
        return samples[:n]

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
