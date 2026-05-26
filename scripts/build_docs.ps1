<#
.SYNOPSIS
  Compile top-level *.tex files to PDF via pdflatex.

.DESCRIPTION
  Runs pdflatex twice per source file so the table of contents resolves on the
  second pass. Output PDFs land in docs/build/. Logs and aux files stay there
  too, so the repository root stays clean.

.PARAMETER File
  Optional .tex filename (relative to repo root). If omitted, builds workflow.tex
  and quickstart.tex.

.EXAMPLE
  pwsh scripts/build_docs.ps1
  pwsh scripts/build_docs.ps1 -File workflow.tex
#>
param(
    [string]$File
)

$ErrorActionPreference = "Stop"
$root  = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root "docs\build"

if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    Write-Error "pdflatex not found in PATH. Install TeX Live or MiKTeX first."
}

New-Item -ItemType Directory -Force -Path $build | Out-Null

$targets = if ($File) {
    @($File)
} else {
    @("workflow.tex", "quickstart.tex")
}

foreach ($t in $targets) {
    $src = Join-Path $root $t
    if (-not (Test-Path $src)) {
        Write-Warning "Skip: $src not found"
        continue
    }
    Write-Host "==> Building $t (pass 1/2)..."
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory $build $src | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pdflatex pass 1 failed for $t. See $build\$([IO.Path]::ChangeExtension($t,'log'))"
    }
    Write-Host "==> Building $t (pass 2/2)..."
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory $build $src | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pdflatex pass 2 failed for $t. See $build\$([IO.Path]::ChangeExtension($t,'log'))"
    }
    $pdf = Join-Path $build ([IO.Path]::ChangeExtension($t, "pdf"))
    Write-Host "    -> $pdf"
}

Write-Host "Done."
