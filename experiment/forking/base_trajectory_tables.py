"""Emit two LaTeX longtables of FULL forking base trajectories for the paper:

  1. paper/sections/table_forking_trajectories.tex -- one row per Qwen3.5 thinking model
     (0.8/2/4/9B) with its OWN full racismo chain of thought; the change-point (forking) tokens
     are highlighted in red.
  2. paper/sections/table_base_trajectories.tex -- one row per bias category with the FULL shared
     Qwen3.5-27B chain of thought that every smaller model re-walks.

  uv run python -m experiment.forking.base_trajectory_tables
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_FORKING = REPO / "out" / "forking"
_SHARED = REPO / "cloud" / "shared_bases"
_OUT = REPO / "paper" / "sections"

_MODELS = [("Qwen3.5-0.8B", "Qwen3.5 0.8B"), ("Qwen3.5-2B", "Qwen3.5 2B"),
           ("Qwen3.5-4B", "Qwen3.5 4B"), ("Qwen3.5-9B", "Qwen3.5 9B")]
_CATS = [("racismo", "racismo"), ("xenofobia", "xenofobia"),
         ("clasismo", "clasismo"), ("genero", "g\\'enero")]


def _esc(s: str) -> str:
    """Escape LaTeX specials and collapse whitespace to a flowing paragraph."""
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def _changepoint_indices(model_dir: Path) -> list[int]:
    """The model's forking-token positions (highest, well-separated peaks of the change-point
    location posterior tau, taking the MAP number of change points)."""
    cp = json.load((model_dir / "forking_analysis.json").open())["change_points"]
    tau = np.array(cp.get("tau_posterior", []), float)
    if not len(tau):
        return []
    ncp = cp.get("num_changepoints_posterior")
    k = int(np.argmax(ncp)) if ncp else (1 if cp.get("significant") else 0)
    if k <= 0:
        return []
    peaks = [(tau[i], i) for i in range(len(tau)) if tau[i] > 0.4
             and (i == 0 or tau[i] >= tau[i - 1]) and (i == len(tau) - 1 or tau[i] >= tau[i + 1])]
    peaks.sort(reverse=True)
    chosen: list[int] = []
    for _h, i in peaks:
        if all(abs(i - j) > 5 for j in chosen):
            chosen.append(i)
        if len(chosen) >= k:
            break
    return sorted(chosen)


def _highlighted(tokens: list[str], forks: set[int]) -> str:
    """Escaped trajectory text with the forking tokens wrapped red-and-bold."""
    out = []
    for i, tok in enumerate(tokens):
        e = _esc(tok)
        if not e:
            e = " "
        out.append(f"\\textcolor{{red}}{{\\textbf{{{e}}}}}" if i in forks else e)
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def _longtable(caption: str, label: str, head: tuple[str, str], rows: list[tuple[str, str]]) -> str:
    lines = [r"\begingroup\scriptsize",
             r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{1.9cm} "
             r">{\raggedright\arraybackslash}p{13.4cm}@{}}",
             f"  \\caption{{{caption}}}\\label{{{label}}}\\\\",
             r"  \toprule", f"  \\textbf{{{head[0]}}} & \\textbf{{{head[1]}}} \\\\", r"  \midrule",
             r"  \endfirsthead",
             r"  \multicolumn{2}{@{}l}{\emph{\tablename~\thetable\ (continued)}}\\ \toprule",
             f"  \\textbf{{{head[0]}}} & \\textbf{{{head[1]}}} \\\\ \\midrule", r"  \endhead",
             r"  \bottomrule", r"  \endfoot"]
    for a, b in rows:
        lines.append(f"  {a} & {b} \\\\[2pt]")
    lines += [r"\end{longtable}", r"\endgroup"]
    return "\n".join(lines) + "\n"


def build() -> None:
    # 1. Per-model full racismo trajectories, forking tokens in red.
    rows = []
    for d, disp in _MODELS:
        t = json.load((_FORKING / d / "forking_trajectory.json").open())
        forks = set(_changepoint_indices(_FORKING / d))
        rows.append((f"\\texttt{{{disp}}}", _highlighted(t["base_token_texts"], forks)))
    (_OUT / "table_forking_trajectories.tex").write_text(_longtable(
        "Per-model chain of thought on the racismo item (\\Cref{fig:forking-grid}; the forked "
        "base path is length-capped), with each model's \\textcolor{red}{\\textbf{forking tokens}} "
        "(change points) in red.",
        "tab:forking-trajectories", ("Model", "Chain of thought (forking tokens in red)"), rows))

    # 2. Full shared 27B base trajectories, one per category.
    rows = []
    for cat, disp in _CATS:
        txt = json.load((_SHARED / f"{cat}.json").open())["base_path_text"]
        rows.append((f"\\texttt{{{disp}}}", _esc(txt)))
    (_OUT / "table_base_trajectories.tex").write_text(_longtable(
        "The four shared \\texttt{Qwen3.5-27B} base trajectories re-walked by every model in "
        "\\Cref{fig:forking-shared-grid}; each reasons to the gold abstention.",
        "tab:base-trajectories", ("Category", "Full \\texttt{Qwen3.5-27B} chain of thought"), rows))
    print("wrote table_forking_trajectories.tex + table_base_trajectories.tex")


if __name__ == "__main__":
    build()
