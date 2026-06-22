#!/usr/bin/env bash
# Build the hackathon report -> build/main.pdf
# Usage: bash build.sh          (full build with bibliography)
#        bash build.sh fast     (single pass, skips bibtex -- faster iteration)
set -euo pipefail

SRCDIR="$(cd "$(dirname "$0")" && pwd)"
BUILDDIR="$SRCDIR/build"
JOB="main"
mkdir -p "$BUILDDIR"

pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILDDIR" "$JOB.tex"

if [ "${1:-}" != "fast" ]; then
  ( cd "$BUILDDIR" && BIBINPUTS="$SRCDIR:" bibtex "$JOB" ) || true
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILDDIR" "$JOB.tex"
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILDDIR" "$JOB.tex"
fi

echo "=== Done: $BUILDDIR/$JOB.pdf ==="
