# SESGO Module

Loads the **SESGO** social-bias benchmark prompts into typed `BaseSchema`
objects. Each item is a (context, question) with three role-labelled answer
options (TARGET / OTHER / UNKNOWN). Both context conditions are loaded; for now
only Spanish **ORIGINAL** rows are kept (English and BBQ-adapted are disabled).

## Structure

```
sesgo/
├── sesgo_category.py   # SesgoCategory enum (file stems + EN labels + from_data_value)
├── sesgo_label.py      # SesgoLabel enum (TARGET / OTHER / UNKNOWN) + from_answer_index
├── sesgo_item.py       # SesgoItem dataclass (.options_in_canonical_order, .context_condition)
└── sesgo_loader.py     # load_items(): glob xlsx, keep es+original, parse answer_info
```

## On-disk layout

```
<sesgo_dir>/prompts/prompts_<cat>_<lang>.xlsx
  cat  ∈ {racismo, xenofobia, clasismo, genero}
  lang ∈ {es, en}
```

Each file is a single flat sheet sharing one schema. **Filename casing is
inconsistent** (`prompts_genero_EN.xlsx` vs `prompts_racismo_en.xlsx`), so the
loader globs case-insensitively.

## Filtering: Spanish + original only (for now)

The loader keeps only **original** rows (`bbq == False`); BBQ-adapted rows are
dropped at read time. `load_items` defaults to **Spanish only** (`languages=("es",)`)
— English is disabled for now. Both axes are intentional, current-scope filters,
not corpus limits.

## Both context conditions (ambig + disambig)

Both `context_condition == "ambig"` and `"disambig"` rows are loaded; the field
is carried on `SesgoItem`. The condition fixes the gold answer:

- **AMBIGUOUS** — the text gives no basis to pick a person, so the correct
  answer is always **UNKNOWN** (`gold_label = UNKNOWN`).
- **DISAMBIGUATED** — the context names a ground-truth role; the data's `label`
  index (0=other, 1=target, 2=unknown) is decoded via
  `SesgoLabel.from_answer_index(label)` into `gold_label`.

## Three-way label semantics

`answer_info` is a stringified Python dict parsed with `ast.literal_eval`. Its
positions encode fixed roles (invariant across the corpus):

| Position | Role     | Meaning                                   |
|----------|----------|-------------------------------------------|
| `ans0`   | OTHER    | the contrasting (non-target) group        |
| `ans1`   | TARGET   | the stereotyped / probed group            |
| `ans2`   | UNKNOWN  | "not enough information" — gold for ambig  |

## Polarity

Each item has `polarity` ∈ {`neg`, `nonneg`}: the negative phrasing asks who fits
the stereotype, the non-negative asks the opposite. The two phrasings of one
scenario share a `question_id` (a stable hash of category+language+context).

## Bias-score note

Because the gold answer is always UNKNOWN, any time the model instead picks
TARGET or OTHER it reveals a directional bias. Comparing TARGET-vs-OTHER pick
rates across the `neg`/`nonneg` polarity pair yields a BBQ-style bias score; this
module only loads the data, it does not compute the score.

## Public API

| Symbol | Purpose |
|--------|---------|
| `SesgoCategory` | Enum of bias categories; `.english`, `.from_data_value(s)` |
| `SesgoLabel` | Enum of answer roles: `TARGET`, `OTHER`, `UNKNOWN`; `.from_answer_index(i)` |
| `SesgoItem` | One prompt; `.options_in_canonical_order`, `.context_condition`, `.gold_label` |
| `load_items(sesgo_dir, categories, languages, limit)` | Glob xlsx (es+original, both conditions) -> item list |

## See Also

- [EXPLANATION.md](./EXPLANATION.md) — API reference and design notes
- [Root CLAUDE.md](../../../CLAUDE.md) — global project rules
