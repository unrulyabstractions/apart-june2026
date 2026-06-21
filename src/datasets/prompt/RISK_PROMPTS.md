# Risk-Assessment Prompt Generation

Turns each `MentalRiskSubject` into a **grid** of risk-assessment prompts. Every
cell is a self-describing `RiskPromptSample`: the rendered prompt string plus the
metadata a downstream querier needs to interpret the model's reply.

## Axes (the grid)

Per subject, the cross product of:

| Axis | Values | Notes |
|------|--------|-------|
| **Framing** | `at_risk_of`, `suffering`, `safe`, `intervene` | Each supplies a question + positive (high-risk) and negative (low-risk) answer phrases in EN/ES. `safe` flips polarity: its high-risk answer is "not safe". |
| **Language** | `en`, `es` | Controls instruction, question, and label phrasing. The transcript stays in its original source language. |
| **Task type** | `SCORE`, `CATEGORIZE` | Different downstream parsing. |
| → SCORE direction | `scale_high` (1 = at risk), `scale_low` (0 = at risk) | 2 variants. The direction is stated in the question. |
| → CATEGORIZE label style | `("a)","b)")`, `("1)","2)")`, `("Yes","No")` | First chars are distinct so a runner can score a single token. |
| → CATEGORIZE order flip | `False`, `True` | Whether the negative phrase is listed first; `positive_idx` records which label index is the at-risk answer. |

Full grid per subject = 4 framings × 2 languages × (2 SCORE + 3×2 CATEGORIZE) =
4 × 2 × 8 = **64 prompts**.

## Rendered format

**CATEGORIZE**

```
<instruction(lang)>

<transcript>
{transcript}
</transcript>

<question(lang)>
{labelA} {phraseA}
{labelB} {phraseB}
{choice_prefix}
```

`labels` = `(labelA, labelB)` style tokens; `phraseA/phraseB` are the framing's
positive/negative phrases in `lang` (order set by the flip); `positive_idx`
records which of A/B is the at-risk answer. A querier calls
`runner.choose(text, choice_prefix, labels)` and compares the chosen index to
`positive_idx`.

**SCORE**

```
<instruction(lang, SCORE)>

<transcript>
{transcript}
</transcript>

<question(lang)> <number cue>, <scale direction hint>.
```

No labels; `scale_high` records whether 1 or 0 is the at-risk end so a parsed
number can be oriented.

## `RiskPromptSample` fields

| Field | Meaning |
|-------|---------|
| `sample_idx` | Position in the dataset. |
| `text` | Fully rendered prompt. |
| `subject_id`, `disorder` | Source provenance. |
| `gold_risk` | Subject's true risk in [0,1] (or `None`). |
| `framing` | Framing key. |
| `task_type` | `RiskTaskType.SCORE` / `CATEGORIZE`. |
| `language` | `en` / `es`. |
| `labels` | `(labelA, labelB)` (categorize only). |
| `positive_idx` | Label index = at-risk answer (categorize only). |
| `choice_prefix` | Answer cue, e.g. `"Answer: "`. |
| `scale_high` | `True` if 1 = at risk (score only). |
| `label_flipped` | Whether option order was flipped. |
| `positive_label` / `negative_label` | Properties: the at-risk / not-at-risk label token. |

## Usage

```python
from src.datasets.mental_risk import load_subjects
from src.datasets.prompt import RiskPromptGenerator, RiskPromptConfig

subjects = load_subjects("path/to/extracted")
dataset = RiskPromptGenerator(RiskPromptConfig(name="my_run")).generate(subjects)
dataset.save_as_json("risk_prompts.json")
```

`RiskPromptConfig` defaults to the full grid; narrow any axis by passing explicit
value lists (`framings`, `task_types`, `label_styles`, `languages`,
`order_flips`, `scale_highs`). The dataset and every sample round-trip through
`to_dict`/`from_dict` and `save_as_json`/`from_json` via `BaseSchema`.

## Modules

| File | Responsibility |
|------|----------------|
| `risk_language.py` | `RiskLanguage` enum (`en`, `es`). |
| `risk_task_type.py` | `RiskTaskType` enum (`SCORE`, `CATEGORIZE`). |
| `risk_framing.py` | `RiskFraming` + `RISK_FRAMINGS` + `get_framing`; EN/ES text & polarity. |
| `risk_label_style.py` | Categorize label-token pairs (reuses `formatting_variation`). |
| `risk_instruction.py` | EN/ES instruction templates + `render_instruction`. |
| `risk_prompt_sample.py` | `RiskPromptSample`. |
| `risk_prompt_config.py` | `RiskPromptConfig` (grid selection + provenance). |
| `risk_prompt_dataset.py` | `RiskPromptDataset` (save/load). |
| `risk_prompt_generator.py` | `RiskPromptGenerator.generate`. |
