"""Family / size / display-name for a stability output dir (`<bare-model>-<mode>`), shared
by every Stage-1 figure so they group by family and order by scale identically.

  Gemma   gemma-4-{E2B->2, E4B->4, 12B, 31B}-it
  Qwen    Qwen3.5-{0.8, 2, 4, 9, 27}B            (thinking / nonthinking)
  Mistral Ministral-3-{3, 8, 14}B-{Instruct, Reasoning}
  Llama   Llama-3.x-{1, 3, 8, 70}B-Instruct
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.common import BaseSchema

FAMILY_ORDER = ["Gemma", "Qwen", "Mistral", "Llama"]
# Okabe-Ito family base colours (a per-model shade is derived by size).
FAMILY_COLOR = {"Gemma": "#CC79A7", "Qwen": "#009E73", "Mistral": "#E69F00", "Llama": "#0072B2"}


@dataclass
class SweepModel(BaseSchema):
    dir_name: str
    family: str
    size_b: float
    mode: str          # thinking / nonthinking
    name: str          # versioned display label e.g. "Qwen3.5 4B" / "Mistral 3 3B-R"


def _ver(bare: str, pattern: str) -> str:
    """The family version embedded in the checkpoint name (e.g. 3.5 from Qwen3.5),
    so a label reads "Qwen3.5 4B" rather than the version-less "Qwen 4B"."""
    m = re.search(pattern, bare, re.IGNORECASE)
    return m.group(1) if m else ""


def _size(bare: str) -> float | None:
    m = re.search(r"gemma-4-E?(\d+)B", bare)
    if m:
        return float(m.group(1))
    m = re.search(r"Qwen3\.5-([\d.]+)B", bare)
    if m:
        return float(m.group(1))
    m = re.search(r"Ministral-3-(\d+)B", bare)
    if m:
        return float(m.group(1))
    m = re.search(r"Llama-3\.\d+-(\d+)B", bare)
    return float(m.group(1)) if m else None


def parse_model(dir_name: str) -> SweepModel | None:
    """Parse `<bare>-<mode>` -> SweepModel, or None for an unrecognized family."""
    mode = "thinking" if dir_name.endswith("-thinking") else "nonthinking"
    bare = dir_name[: -len(f"-{mode}")] if dir_name.endswith(f"-{mode}") else dir_name
    low = bare.lower()
    size = _size(bare)
    if size is None:
        return None
    if "gemma" in low:
        fam, name = "Gemma", f"Gemma {_ver(bare, r'gemma-([\d.]+)')} {size:g}B"
    elif "qwen" in low:
        tag = " (think)" if mode == "thinking" else ""
        fam, name = "Qwen", f"Qwen{_ver(bare, r'Qwen([\d.]+)')} {size:g}B{tag}"
    elif "ministral" in low:
        tag = "-R" if "reasoning" in low else "-I"
        fam, name = "Mistral", f"Mistral {_ver(bare, r'Ministral-([\d.]+)')} {size:g}B{tag}"
    elif "llama" in low:
        fam, name = "Llama", f"Llama {_ver(bare, r'Llama-([\d.]+)')} {size:g}B"
    else:
        return None
    return SweepModel(dir_name, fam, size, mode, name)
