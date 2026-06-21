# MentalRisk — Design & API Reference

This module turns the raw, encrypted MentalRiskES corpus into typed objects that
fit the repo's `BaseSchema` conventions, exposing a single continuous risk score
per subject.

## Conceptual Overview

MentalRiskES tracks social-media users over time and asks annotators whether each
subject "suffers" from a disorder. Each subject therefore has:

1. A **message timeline** — an ordered list of posts.
2. A set of **gold label columns** — binary flags (`bs`, `bc`, ...) and annotator
   fractions (`rbs`, `rbc`, ...), where every `r*` column is `count/10`.

Downstream code wants one number, so we collapse the columns into a risk score.

### Risk collapse

```
risk = rbs            if rbs present        # annotator agreement fraction
     = float(bs)      elif bs present       # binary suffers flag
     = None           otherwise             # unlabelled
```

The result is clamped to `[0, 1]`. `None` is preserved (not coerced to `0.0`) so
"unlabelled" is distinguishable from "labelled, zero risk".

## Data Flow

```
encrypted *.zip ──(resolve_password + extract_corpus)──▶ <root>/<source>/<Disorder>/...
                                                              │
                                          load_subjects ──────┘
                                                              │
                                                              ▼
                                               list[MentalRiskSubject]
```

`load_subjects` walks `<extracted>/<source>/<Disorder>/data/subject*.json`,
reads the sibling `gold/gold_label.csv`, and joins them by normalized subject id.
Subject files are sorted numerically (`subject2` before `subject10`). Missing
disorder directories are skipped rather than raising.

## Subject-id normalization

The gold CSV may key subjects as a bare number, a stem, or a filename. All three
collapse to the canonical `subject<N>` via `normalize_subject_id`, so
`"103"`, `"subject103"`, and `"subject103.json"` map identically.

## Key Types

| File | Type | Notes |
|------|------|-------|
| `mental_risk_disorder.py` | `Disorder` | Values = dir names; `.label` / `.label_es` |
| `mental_risk_message.py` | `MentalRiskMessage` | `id_message`, `message`, `date` |
| `mental_risk_subject.py` | `MentalRiskSubject` | `.transcript`, `.risk`, `.n_messages` |

`MentalRiskSubject` round-trips through `to_dict` / `from_dict` (the `Disorder`
enum and nested `MentalRiskMessage` list are handled by `BaseSchema`).

## Gold TXT keying assumption

`read_gold_txt` handles headerless whitespace/comma files where column 0 is the
subject id and column 1 is a single value. Integer values (`0`/`1`) are stored as
`bs` (binary flag); fractional values are stored as `rbs` (annotator fraction).
This mirrors the collapse preference so either form yields the right risk.

## Encryption

`mental_risk_archive.py` uses `pyzipper.AESZipFile` with a password from
`MENTALRISK_ZIP_PASSWORD` (or a password file). `extract_archive` recurses into
nested zips (same password) and deletes them. The password is never hardcoded;
`resolve_password` raises a clear, actionable error when it is absent.
