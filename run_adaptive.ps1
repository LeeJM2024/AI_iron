param(
    [string]$TrainDir = (Join-Path $PSScriptRoot '..\煤气发电预测优化-初赛训练和测试集\初赛-参赛者使用'),
    [string]$TestDir = (Join-Path $PSScriptRoot '..\煤气发电预测优化-初赛训练和测试集\初赛-评分所用测试集'),
    [string]$OutputDir = (Join-Path $PSScriptRoot 'output\adaptive_v10'),
    [switch]$Develop
)
$ErrorActionPreference = 'Stop'
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$pythonExecutable = if (Test-Path -LiteralPath $localPython) { $localPython } else { 'python' }
$trainingPath = (Resolve-Path -LiteralPath $TrainDir).Path
$testingPath = (Resolve-Path -LiteralPath $TestDir).Path
Push-Location -LiteralPath $PSScriptRoot
try {
    if ($Develop) {
        & $pythonExecutable -X utf8 -W ignore::DeprecationWarning develop_compact.py --train-dir $trainingPath
        if ($LASTEXITCODE -ne 0) { throw 'Base development failed' }
        & $pythonExecutable -X utf8 -W ignore::DeprecationWarning develop_online.py --train-dir $trainingPath
        if ($LASTEXITCODE -ne 0) { throw 'Online development failed' }
        & $pythonExecutable -X utf8 -W ignore::DeprecationWarning relative_linear.py --train-dir $trainingPath
        if ($LASTEXITCODE -ne 0) { throw 'Relative-error development failed' }
        & $pythonExecutable -X utf8 -W ignore::DeprecationWarning select_online.py --train-dir $trainingPath --include-relative --output-dir artifacts/adaptive_selected
        if ($LASTEXITCODE -ne 0) { throw 'Selection failed' }
    }
    & $pythonExecutable -X utf8 -W ignore::DeprecationWarning run_prelim.py --train-dir $trainingPath --test-dir $testingPath --selection artifacts/adaptive_selected/selection.json --output-dir $OutputDir
    if ($LASTEXITCODE -ne 0) { throw 'Generation or verification failed; do not submit an old archive' }
} finally { Pop-Location }
