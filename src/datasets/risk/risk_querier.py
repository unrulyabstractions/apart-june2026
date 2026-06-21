"""Run risk prompts through a local LLM at two levels and record the results.

Level 1 (non-thinking): teacher-force the two answer options and read a
calibrated P(at risk) from their logprobs — fast and reasoning-free.
Level 2 (thinking): sample several free-form generations, parse a risk score
from each, and summarize the cloud. CATEGORIZE uses the prompt's own labels;
SCORE synthesizes scale-anchor labels so the same machinery yields a
probability for numeric prompts too.
"""

from __future__ import annotations

import math

from src.binary_choice.binary_choice_runner import BinaryChoiceRunner
from src.common.device_utils import ProgressTracker, clear_gpu_memory
from src.common.logging import log
from src.datasets.prompt import RiskPromptDataset, RiskPromptSample, RiskTaskType
from .non_thinking_result import NonThinkingResult
from .risk_assessment_sample import RiskAssessmentSample
from .risk_dataset import RiskDataset
from .risk_query_config import RiskQueryConfig
from .risk_response_parsing import parse_risk_score
from .score_summary import summarize_scores

# Periodically free accelerator memory so long runs don't accumulate caches.
_GPU_CLEAR_EVERY = 25


def _calibrated_risk(lp_pos: float, lp_neg: float) -> float:
    """Numerically-stable 2-way softmax P(at-risk option)."""
    m = max(lp_pos, lp_neg)
    e_pos, e_neg = math.exp(lp_pos - m), math.exp(lp_neg - m)
    return e_pos / (e_pos + e_neg)


class RiskQuerier:
    """Queries a BinaryChoiceRunner for both risk-assessment levels."""

    def __init__(self, config: RiskQueryConfig):
        self.config = config
        self._runner: BinaryChoiceRunner | None = None

    def _load_model(self, name: str) -> BinaryChoiceRunner:
        """Construct (and cache) the runner; reuse it across samples/datasets."""
        if self._runner is not None and self._runner.model_name == name:
            return self._runner
        self._runner = BinaryChoiceRunner(model_name=name)
        return self._runner

    def _non_thinking(
        self, sample: RiskPromptSample, runner: BinaryChoiceRunner
    ) -> NonThinkingResult:
        """Teacher-forced calibrated risk for one CATEGORIZE prompt."""
        labels, pos_idx = sample.labels, sample.positive_idx
        choice = runner.choose(sample.text, sample.choice_prefix or "Answer: ", labels)
        logprobs = choice.divergent_logprobs
        lp_pos, lp_neg = logprobs[pos_idx], logprobs[1 - pos_idx]
        return NonThinkingResult(
            predicted_risk=_calibrated_risk(lp_pos, lp_neg),
            choice_idx=choice.choice_idx,
            logprob_positive=lp_pos,
            logprob_negative=lp_neg,
            positive_label=labels[pos_idx],
            negative_label=labels[1 - pos_idx],
        )

    def _thinking_generations(
        self, sample: RiskPromptSample, runner: BinaryChoiceRunner
    ) -> list[str]:
        """Sample N free-form reasoning completions for one prompt."""
        return [
            runner.generate(
                sample.text,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
            for _ in range(self.config.n_thinking_samples)
        ]

    def query_sample(
        self, prompt_sample: RiskPromptSample, runner: BinaryChoiceRunner
    ) -> RiskAssessmentSample:
        """Run both enabled levels for one prompt and assemble the record.

        Non-thinking is a binary teacher-forced readout, so it only applies to
        CATEGORIZE prompts; SCORE prompts ask for a free number (whose first
        token, e.g. the "0" of "0.75", is uninformative as a binary anchor) and
        are covered by the thinking level alone.
        """
        non_thinking = (
            self._non_thinking(prompt_sample, runner)
            if self.config.do_non_thinking
            and prompt_sample.task_type is RiskTaskType.CATEGORIZE
            else None
        )

        thinking = completions = None
        if self.config.do_thinking:
            completions = self._thinking_generations(prompt_sample, runner)
            scores = [
                s
                for s in (parse_risk_score(g, prompt_sample) for g in completions)
                if s is not None
            ]
            thinking = summarize_scores(scores)

        return RiskAssessmentSample(
            sample_idx=prompt_sample.sample_idx,
            subject_id=prompt_sample.subject_id,
            disorder=prompt_sample.disorder,
            framing=prompt_sample.framing,
            language=prompt_sample.language,
            task_type=prompt_sample.task_type,
            gold_risk=prompt_sample.gold_risk,
            prompt_text=prompt_sample.text,
            non_thinking=non_thinking,
            thinking=thinking,
            _thinking_completions=completions,
        )

    def _subsample(self, samples: list[RiskPromptSample]) -> list[RiskPromptSample]:
        """Deterministically keep the first ``subsample`` fraction of prompts.

        A leading slice (not a random sample) keeps runs reproducible without a
        seed and preserves the per-subject prompt grouping in dataset order.
        """
        frac = self.config.subsample
        if frac >= 1.0 or not samples:
            return samples
        n = max(1, math.ceil(len(samples) * frac))
        return samples[:n]

    def query_dataset(
        self, prompt_dataset: RiskPromptDataset, model_name: str
    ) -> RiskDataset:
        """Query every prompt in a dataset and collect a RiskDataset."""
        runner = self._load_model(model_name)
        samples = self._subsample(prompt_dataset.samples)
        log(f"[risk] Querying {len(samples)} prompts with {runner.model_name}...")

        tracker = ProgressTracker(total=len(samples), progress_every=10, memory_every=50)
        results: list[RiskAssessmentSample] = []
        for i, prompt_sample in enumerate(samples):
            tracker.step(i)
            results.append(self.query_sample(prompt_sample, runner))
            if (i + 1) % _GPU_CLEAR_EVERY == 0:
                clear_gpu_memory()
        clear_gpu_memory()

        return RiskDataset(
            prompt_dataset_id=prompt_dataset.dataset_id,
            model=model_name,
            config=self.config,
            samples=results,
        )
