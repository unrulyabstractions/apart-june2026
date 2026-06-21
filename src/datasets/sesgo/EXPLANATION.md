# SESGO — Design & API Reference

This module turns SESGO's Excel prompt files into typed `SesgoItem` objects that
fit the repo's `BaseSchema` conventions, exposing the three answer options with
their fixed roles so downstream bias scoring needs no knowledge of the on-disk
column scheme.

## Conceptual Overview

SESGO is a BBQ-style social-bias benchmark across four categories (racism,
xenophobia, classism, gender) in Spanish and English. Each question presents a
short context and three options. In an **ambiguous** context the context alone
does not justify any person, so the only correct answer is "not enough
information" (UNKNOWN). A model that instead picks a *person* exposes a bias; the
direction (TARGET vs OTHER) and its flip across the `neg`/`nonneg` polarity pair
is what a bias score measures. This module loads only the ambiguous rows.

## Data Flow

```
prompts/prompts_<cat>_<lang>.xlsx ──(_find_prompt_file: case-insensitive glob)──▶ path
                                                                                   │
                          _load_file: read_excel + filter context_condition=="ambig"
                                                                                   │
                          _row_to_item: ast.literal_eval(answer_info) -> roles     │
                                                                                   ▼
                                                                    list[SesgoItem]
```

`load_items` iterates the requested (category, language) pairs, locates each
file case-insensitively, keeps only `context_condition == "ambig"`, and unpacks
`answer_info`. Missing files are logged and skipped rather than raising.

## The positional convention

`answer_info` is a stringified Python dict, e.g.
`{"ans0": "...", "ans1": "...", "ans2": "..."}`. The positions are an **invariant
role encoding**: `ans0` = OTHER, `ans1` = TARGET, `ans2` = UNKNOWN. We unpack
these into `other_text` / `target_text` / `unknown_text` once, at load time, so
no downstream code re-derives roles from positions. `options_in_canonical_order`
re-emits them in the original ans0/ans1/ans2 order as role-tagged pairs.

## question_id and polarity pairing

A scenario appears twice — once negatively ("who was lazy?") and once
non-negatively ("who was hardworking?"). Both rows share context, category and
language and differ only in `question_polarity` and `question`. `question_id` is
`blake2b(category + language + context)`, so the pair collapses to one id and
callers can recover the polarity pair from a flattened list.

## Why not trust the `category` column

The corpus' own `category` cells are inconsistent across files — observed values
include `Racism`, `Xenophoby`, lowercase `gender`, Spanish `clasismo`, and `SES`.
The enum is therefore keyed off the **filename stem** (the one reliable
identifier), and `SesgoCategory.from_data_value` is alias- and case-tolerant so a
raw cell can still be mapped when needed.

## Key Types

| File | Type | Notes |
|------|------|-------|
| `sesgo_category.py` | `SesgoCategory` | Values = file stems; `.english`, `.from_data_value` |
| `sesgo_label.py` | `SesgoLabel` | `TARGET`, `OTHER`, `UNKNOWN` |
| `sesgo_item.py` | `SesgoItem` | `.options_in_canonical_order`; gold defaults UNKNOWN |

`SesgoItem` round-trips through `to_dict` / `from_dict` (the `SesgoCategory` and
`SesgoLabel` enums are handled by `BaseSchema`).

## Testing

A small synthetic fixture (`tests/fixtures/sesgo/`) of invented prompts — no real
SESGO content — drives `tests/unit/datasets/test_sesgo_loader.py`, covering the
ambig filter, role unpacking, UNKNOWN gold, polarity-pair id sharing, `limit`,
and the dataclass round-trip. Its mixed-case filename also exercises the
case-insensitive glob.
