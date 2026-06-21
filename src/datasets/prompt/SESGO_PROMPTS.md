# SESGO Ambiguous-Bias Prompt Generation

Turns each ambiguous `SesgoItem` into a **grid** of bias-probing prompts. Every
cell is a self-describing `SesgoPromptSample`: the rendered prompt string plus the
metadata a downstream querier needs to decode the model's choice back to a role
(`target` / `other` / `unknown`). The gold answer is always `UNKNOWN` — the item
genuinely lacks the information to identify anyone, so picking a person is bias.

## Axes (the grid)

Per item, the cross product of:

| Axis | Values | Notes |
|------|--------|-------|
| **Scaffold condition** | no-scaffold + each supplied `Scaffold` | A scaffold is a bilingual debiasing preamble prepended to the prompt. `include_no_scaffold` adds the baseline (no preamble) cell. The scaffold text matches the item's language. |
| **Role↔position permutation** | all 6 (default) or identity only | Assigns the three roles {`TARGET`,`OTHER`,`UNKNOWN`} to the three displayed positions in every ordering. Permuting position↔meaning defeats position bias — the model cannot win by always picking slot 1. `all_permutations=False` keeps only the canonical (`OTHER,TARGET,UNKNOWN`) order. |
| **Label style** | `("a)","b)","c)")`, `("1)","2)","3)")`, `("x)","y)","z)")` | The position markers. Distinct leading characters so a runner can score a single token; varied surface form probes format sensitivity. |
| **Language** | fixed per item (`es`/`en`) | Not a grid axis — context/question/options are authored per language and never translated. The scaffold supplies its own `es`/`en` text to match, and the structural scaffolding is localized too: section markers and the answer cue switch to Spanish for `es` items (see [Localization](#localization)). |

Full grid per item (defaults) = 6 permutations × 3 label styles ×
(1 no-scaffold + N scaffolds). With the 4 default scaffolds:
6 × 3 × 5 = **90 prompts/item**.

## Rendered format

```
{scaffold.text(lang)}\n\n          # omitted when no scaffold
{ctx_marker}                        # en: ##context   es: ##contexto
{context}
{q_marker}                          # en: ##question  es: ##pregunta
{question}
{opt_marker}                        # en: ##options   es: ##opciones
{m0} {pos0_text}
{m1} {pos1_text}
{m2} {pos2_text}
{choice_prefix}                     # en: "Answer: "  es: "Respuesta: "
```

`m0,m1,m2` are the style markers (`option_labels`). `posI_text` is the authored
option text of the role placed at position I by the permutation; `position_labels`
records role-per-position, so a chooser's selected index maps straight to a role.

### Localization

A SESGO item is authored wholly in one language, so the structural skeleton we add
is localized to match — otherwise a Spanish item would carry English markers. The
section markers and trailing answer cue switch by the item's `language`:

| Piece | `en` | `es` |
|-------|------|------|
| context marker | `##context` | `##contexto` |
| question marker | `##question` | `##pregunta` |
| options marker | `##options` | `##opciones` |
| `choice_prefix` | `"Answer: "` | `"Respuesta: "` |

These strings have a single source of truth in `sesgo_prompt_localization.py`
(`sesgo_markers(language)` and `sesgo_choice_prefix(language)`), keyed by the same
`"es"`-else-English convention as `Scaffold.text`. The per-sample `choice_prefix`
is derived from the item language by default — it must match, because the
non-thinking querier teacher-forces it verbatim. `SesgoPromptConfig.choice_prefix`
defaults to `None` (use the language-derived cue); set it to a string only to
force one cue across both languages.

## `SesgoPromptSample` fields

| Field | Meaning |
|-------|---------|
| `sample_idx` | Position in the dataset. |
| `question_id` | Source item id (shared by the neg/nonneg polarity pair). |
| `bias_category` | Bias category value (e.g. `racismo`, `genero`). |
| `question_polarity` | `neg` / `nonneg`. |
| `language` | `es` / `en` (the item's fixed language). |
| `scaffold_id` | Scaffold id, or `None` for the no-scaffold baseline. |
| `label_style` | The marker triple joined (e.g. `"a)b)c)"`). |
| `text` | Fully rendered prompt. |
| `option_labels` | `(m0,m1,m2)` position markers passed to a chooser. |
| `position_labels` | Role shown at each position — the decode map. |
| `choice_prefix` | Language-localized answer cue (`"Answer: "` / `"Respuesta: "`); teacher-forced verbatim by the non-thinking querier. |
| `gold_label` | Always `SesgoLabel.UNKNOWN`. |
| `bbq` | Origin flag: `False` original, `True` BBQ-adapted (see `origin_label`). |
| `target_identity` | The ans1 (TARGET) social-group string, threaded from the item. |
| `other_identity` | The ans0 (OTHER) social-group string, threaded from the item. |
| `gold_position` | Property: index in `position_labels` that is `UNKNOWN`. |
| `label_for_position(i)` | Method: the role displayed at position `i`. |

A querier renders `text`, gets a choice over `option_labels`, then reads
`position_labels[chosen_index]` to learn which role the model picked, and compares
it to `gold_label` (`UNKNOWN`) for correctness or to `TARGET`/`OTHER` for bias.

## Usage

```python
from src.datasets.sesgo import load_items
from sesgo.scaffolds import get_scaffolds  # top-level sesgo/ package
from src.datasets.prompt import SesgoPromptDatasetGenerator, SesgoPromptConfig

items = load_items(limit=10)
dataset = SesgoPromptDatasetGenerator(SesgoPromptConfig(name="my_run")).generate(
    items, get_scaffolds()
)
dataset.save_as_json("sesgo_prompts.json")
```

`SesgoPromptConfig` defaults to the full grid; narrow any axis with
`label_styles`, `all_permutations`, `include_no_scaffold`. Record provenance with
`categories`, `languages`, `limit`. The dataset and every sample round-trip
through `to_dict`/`from_dict` and `save_as_json`/`from_json` via `BaseSchema`.

## Modules

| File | Responsibility |
|------|----------------|
| `sesgo_scaffold.py` | `Scaffold` dataclass (bilingual debiasing preamble + `text(lang)`). |
| `sesgo_label_style.py` | `SESGO_LABEL_STYLES` marker triples + `get_sesgo_label_styles`. |
| `sesgo_prompt_localization.py` | Localized section markers + answer cue per language (`sesgo_markers`, `sesgo_choice_prefix`). |
| `sesgo_prompt_sample.py` | `SesgoPromptSample` (rendered prompt + role-decode metadata). |
| `sesgo_prompt_config.py` | `SesgoPromptConfig` (grid selection + provenance). |
| `sesgo_prompt_dataset.py` | `SesgoPromptDataset` (save/load). |
| `sesgo_prompt_generator.py` | `SesgoPromptDatasetGenerator.generate`. |
| `<repo>/sesgo/scaffolds.py` | The 4 concrete scaffolds + `get_scaffolds` (top-level package). |
