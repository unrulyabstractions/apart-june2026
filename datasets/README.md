# Datasets (local-only)

External datasets used by this project. **The data itself is git-ignored and
kept local only** — only this manifest is tracked. Re-download each dataset with
the command shown below. Each lands under `datasets/<name>/` with its upstream
`LICENSE`/`README` preserved for attribution.

## `americas_nli/`
- **Source**: https://huggingface.co/datasets/nala-cub/americas_nli
- **What**: AmericasNLI — XNLI (natural language inference) extended to 10
  Indigenous languages of the Americas: Aymara (`aym`), Asháninka (`cni`),
  Bribri (`bzd`), Guaraní (`gn`), Náhuatl (`nah`), Otomí (`oto`), Quechua
  (`quy`), Rarámuri (`tar`), Shipibo-Konibo (`shp`), Wixarika/Huichol (`hch`),
  plus `all_languages`; `validation`/`test` parquet splits.
- **License**: CC BY-SA 4.0.
- **Download**:
  ```bash
  uv run python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='nala-cub/americas_nli', repo_type='dataset', local_dir='datasets/americas_nli')"
  ```

## `corpusMentalRiskES/`
- **Source**: https://github.com/sinai-uja/corpusMentalRiskES
- **What**: MentalRiskES — Spanish dataset for early detection of mental-risk
  disorders (eating disorder, depression, anxiety) from pseudonymized Telegram
  threads; IberLEF MentalRiskES 2023/2024/2025 editions.
- **License**: CC BY-NC-SA 4.0 — attribution, **non-commercial**, share-alike.
- **⚠️ Sensitivity**: sensitive mental-health text; upstream states
  **non-clinical research only** and points to an access-request form. Kept
  local-only (git-ignored) by design — do not commit the data to this repo.
- **Download**:
  ```bash
  git clone https://github.com/sinai-uja/corpusMentalRiskES datasets/corpusMentalRiskES
  ```

## `SESGO/`
- **Source**: https://github.com/mvrobles/SESGO
- **What**: SESGO — bias-evaluation prompts/templates for LLMs in Spanish and
  English across racism, xenophobia, classism, and gender, with results and
  analysis notebooks.
- **License**: see upstream repo (no explicit LICENSE file at time of fetch).
- **Download**:
  ```bash
  git clone https://github.com/mvrobles/SESGO datasets/SESGO
  ```
