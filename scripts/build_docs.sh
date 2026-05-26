#!/usr/bin/env bash
# Compile top-level *.tex files to PDF via pdflatex.
# Runs pdflatex twice per file so TOC resolves on second pass.
# Output PDFs land in docs/build/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/docs/build"

if ! command -v pdflatex >/dev/null 2>&1; then
    echo "ERROR: pdflatex not in PATH. Install TeX Live or MiKTeX." >&2
    exit 1
fi

mkdir -p "$BUILD"

if [[ $# -gt 0 ]]; then
    targets=("$@")
else
    targets=("workflow.tex" "quickstart.tex")
fi

for t in "${targets[@]}"; do
    src="$ROOT/$t"
    if [[ ! -f "$src" ]]; then
        echo "WARN: $src not found, skipping" >&2
        continue
    fi
    echo "==> Building $t (pass 1/2)..."
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$BUILD" "$src" >/dev/null
    echo "==> Building $t (pass 2/2)..."
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$BUILD" "$src" >/dev/null
    echo "    -> $BUILD/${t%.tex}.pdf"
done

echo "Done."
