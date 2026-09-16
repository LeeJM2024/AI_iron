param([string]$OutputDir = (Join-Path $PSScriptRoot 'output\current'))
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    python -X utf8 current_release.py --output-dir $OutputDir
    if ($LASTEXITCODE -ne 0) { throw 'Current release verification/export failed' }
} finally { Pop-Location }
