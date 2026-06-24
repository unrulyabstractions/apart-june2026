"""Main-paper figure: what language does each THINKING model reason in?

The SESGO prompts are Spanish, yet the chain of thought is not always: Qwen3.5 thinking
reasons in English ("Thinking Process: 1. Analyze the Request..."), while Ministral-3
Reasoning reasons in Spanish ("[THINK]Vamos a analizar el problema..."). We classify each
thinking slice's CoT (the model's own ``response_text``) as Spanish / English / other with
a dependency-free, high-precision function-word + diacritic detector, then draw one stacked
horizontal bar per reasoning model.

  uv run python -m experiment.stability.cot_language_figure --stability-dir out/stability \
      --out-dir paper/figures
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from experiment.common.sweep_models import parse_model

# High-precision markers: function words that are unambiguous to ONE language (shared or
# loan words like "a"/"no"/"contexto-vs-context" are deliberately excluded).
_ES = {"que", "de", "la", "el", "los", "las", "una", "por", "para", "con", "del", "más",
       "pero", "como", "porque", "información", "pregunta", "contexto", "respuesta",
       "vamos", "necesito", "primero", "está", "hay", "problema", "persona", "análisis",
       "entre", "sobre", "debe", "esta", "este", "blanco", "negro", "sus"}
_EN = {"the", "and", "this", "that", "for", "with", "are", "analyze", "request",
       "question", "answer", "process", "options", "option", "enough", "should",
       "because", "need", "who", "context", "information", "white", "black", "step"}
_WORD = re.compile(r"[a-záéíóúñ]+")
_DIACRITIC = re.compile(r"[áéíóúñ¿¡]")
# Quoted spans are usually the model copying the Spanish PROMPT verbatim into its
# reasoning; strip them so we classify the model's OWN prose, not the quoted prompt
# (a verbose English reasoner that quotes the Spanish context heavily would otherwise
# count as Spanish).
_QUOTED = re.compile(r"«[^»]*»|\"[^\"]*\"|“[^”]*”|‘[^’]*’|`[^`]*`")
# The CoT ends at the close-think tag / the Spanish final-answer line; everything after
# (often a degenerate repetition of "Respuesta final: ...") is NOT reasoning and would
# swamp the count with the Spanish answer phrase, so we score only the reasoning prefix.
_COT_END = ("</think>", "[/think]", "respuesta final", "final answer:")


def _cot_only(text: str) -> str:
    low = text.lower()
    ends = [i for i in (low.find(m) for m in _COT_END) if i != -1]
    return text[: min(ends)] if ends else text
_LANGS = ("spanish", "english", "other")
_COLOR = {"spanish": "#D55E00", "english": "#0072B2", "other": "#999999"}
_LABEL = {"spanish": "Spanish", "english": "English", "other": "Other / mixed"}


def detect_cot_language(text: str) -> str:
    """Classify a chain of thought as spanish / english / other by marker dominance,
    after restricting to the reasoning prefix and dropping quoted spans (the copied
    Spanish prompt) so only the model's own reasoning prose is scored."""
    low = _QUOTED.sub(" ", _cot_only(text).lower())
    toks = _WORD.findall(low)
    if len(toks) < 8:
        return "other"
    es = sum(t in _ES for t in toks) + len(_DIACRITIC.findall(low))
    en = sum(t in _EN for t in toks)
    if es == 0 and en == 0:
        return "other"
    if es >= 1.3 * en:
        return "spanish"
    if en >= 1.3 * es:
        return "english"
    return "other"


def _fractions(model_dir: Path) -> tuple[Counter, int]:
    """Count CoT languages over all of one thinking slice's samples."""
    samples = json.load((model_dir / "response_samples.json").open())["samples"]
    c = Counter(detect_cot_language(s["response_text"]) for s in samples)
    return c, len(samples)


def build(stab_root: Path, out_dir: Path) -> Path:
    rows = []  # (name, size, Counter, n)
    for d in sorted(stab_root.iterdir()):
        sm = parse_model(d.name) if d.is_dir() else None
        if sm is None or sm.mode != "thinking" or not (d / "response_samples.json").exists():
            continue
        c, n = _fractions(d)
        rows.append((sm.name, sm.size_b, c, n))
    rows.sort(key=lambda r: r[1])  # by model size

    fig, ax = plt.subplots(figsize=(9, 0.7 * len(rows) + 1.6))
    ypos = range(len(rows))
    for i, (_name, _size, c, n) in enumerate(rows):
        left = 0.0
        for lang in _LANGS:
            frac = c.get(lang, 0) / n
            if frac <= 0:
                continue
            ax.barh(i, frac, left=left, color=_COLOR[lang], edgecolor="white", height=0.62)
            if frac >= 0.06:  # label only segments wide enough to hold the %
                ax.text(left + frac / 2, i, f"{frac*100:.0f}%", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            left += frac
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([f"{r[0]}  (n={r[3]})" for r in rows], fontsize=10)
    ax.set_xlim(0, 1); ax.set_xlabel("Share of chain-of-thought traces")
    ax.set_title("Reasoning-trace language by model\n(Spanish prompts; which language is the chain of thought?)",
                 fontsize=12, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=_COLOR[l]) for l in _LANGS]
    ax.legend(handles, [_LABEL[l] for l in _LANGS], loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cot_language_by_model.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{'model':22s} {'n':>6s}  spanish  english  other")
    for name, _size, c, n in rows:
        print(f"{name:22s} {n:6d}  {c.get('spanish',0)/n:7.2%}  {c.get('english',0)/n:7.2%}  {c.get('other',0)/n:6.2%}")
    print(f"\nWrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stability-dir", type=Path, default=Path("out/stability"))
    ap.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    a = ap.parse_args()
    build(a.stability_dir, a.out_dir)


if __name__ == "__main__":
    main()
