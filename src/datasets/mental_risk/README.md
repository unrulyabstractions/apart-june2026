# MentalRisk Module

Loads the [MentalRiskES](https://sites.google.com/view/mentalriskes) corpus into
typed `BaseSchema` objects: subjects with ordered message timelines and a single
collapsed risk score in `[0, 1]`.

## Structure

```
mental_risk/
├── mental_risk_disorder.py     # Disorder enum (dir names + EN/ES labels)
├── mental_risk_message.py      # MentalRiskMessage dataclass
├── mental_risk_subject.py      # MentalRiskSubject dataclass (transcript/risk)
├── risk_label_collapse.py      # collapse_risk(): gold columns -> [0, 1] score
├── mental_risk_gold.py         # read_gold_csv / read_gold_txt + id normalizer
├── mental_risk_archive.py      # encrypted-zip extraction (pyzipper)
└── mental_risk_loader.py       # load_subjects(): walk extracted tree
```

## On-disk layout (after extraction)

```
<root>/<source>/<Disorder>/data/subjectN.json   # source ∈ {processed, raw}
<root>/<source>/<Disorder>/gold/gold_label.csv  # Disorder ∈ {Anxiety, Depress, ED}
```

`subjectN.json` is a JSON array of `{id_message, message, date}`. The gold CSV
has one row per subject; column casing is read from the file's header at load
time. Anxiety columns: `bs,bc,rbs,rbc`; Depress/ED add `bsf,bsa,bso,rsf,rsa,rso,rc`.

## Risk collapse rule

`collapse_risk` reduces the gold columns to one score in `[0, 1]`: prefer `rbs`
(fraction of annotators marking "suffers"), fall back to `bs`, else `None`.
Values are clamped to `[0, 1]`.

## Encrypted data caveat

The official archives are **ZipCrypto/AES encrypted**. No corpus data ships in
this repo — only a small synthetic fixture under `tests/fixtures/mental_risk/`.
The password is read from the `MENTALRISK_ZIP_PASSWORD` environment variable (or
a password file); it is **never** hardcoded. Obtain it via the corpus access
agreement, then call `extract_corpus`.

## Public API

| Symbol | Purpose |
|--------|---------|
| `Disorder` | Enum of disorders; `.label` (EN) / `.label_es` (ES) phrasings |
| `MentalRiskMessage` | One `{id_message, message, date}` record |
| `MentalRiskSubject` | Subject with `.transcript`, `.risk`, `.n_messages` |
| `collapse_risk(labels)` | Gold dict -> single risk score or `None` |
| `read_gold_csv(path)` / `read_gold_txt(path)` | Gold readers keyed by subject id |
| `load_subjects(extracted_dir, ...)` | Walk extracted tree -> subject list |
| `extract_corpus(corpus_dir, out_dir, password)` | Decrypt + unzip archives |
| `resolve_password(password_file=None)` | Password from env/file |

## See Also

- [EXPLANATION.md](./EXPLANATION.md) — API reference and design notes
- [Root CLAUDE.md](../../../CLAUDE.md) — global project rules
