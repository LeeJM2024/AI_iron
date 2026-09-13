param(
    [string]$TrainDir = (Join-Path $PSScriptRoot '..\煤气发电预测优化-初赛训练和测试集\初赛-参赛者使用'),
    [string]$TestDir = (Join-Path $PSScriptRoot '..\煤气发电预测优化-初赛训练和测试集\初赛-评分所用测试集'),
    [string]$OutputDir = (Join-Path $PSScriptRoot 'output\pooled_v4'),
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
        & $pythonExecutable -X utf8 -W ignore::DeprecationWarning develop_stable.py --train-dir $trainingPath --robust-raw --output-dir artifacts/stable_v3_robust_linear
        if ($LASTEXITCODE -ne 0) { throw 'Base-model historical development failed' }
        & $pythonExecutable -X utf8 -W ignore::DeprecationWarning pooled_short.py --train-dir $trainingPath
        if ($LASTEXITCODE -ne 0) { throw 'Pooled-model historical development failed' }
        & $pythonExecutable -X utf8 -W ignore::DeprecationWarning select_pooled.py --train-dir $trainingPath
        if ($LASTEXITCODE -ne 0) { throw 'Pooled-model selection failed' }
    }
    & $pythonExecutable -X utf8 -W ignore::DeprecationWarning run_prelim.py --train-dir $trainingPath --test-dir $testingPath --selection artifacts/pooled_v4_selected/selection.json --output-dir $OutputDir
    if ($LASTEXITCODE -ne 0) { throw 'Generation/verification failed; do not submit an old archive' }
} finally { Pop-Location }
