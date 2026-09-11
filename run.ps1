$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ScriptDir
try {
    if (!(Test-Path -LiteralPath "artifacts/development/selection.json")) {
        python -X utf8 experiment.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    python -X utf8 main.py --official --input-dir "$ScriptDir" --output-dir "$ScriptDir/output/official" @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
