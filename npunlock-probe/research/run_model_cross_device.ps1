param([string]$XpuPython)
$ErrorActionPreference = 'Stop'
if (-not $XpuPython) { throw 'Specify -XpuPython with the path to your Intel GPU PyTorch python.exe' }
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$runner = Join-Path $PSScriptRoot 'model_cross_device.py'
$deps = Join-Path $PSScriptRoot 'model-benchmark-deps'
if (!(Test-Path -LiteralPath $XpuPython) -or !(Test-Path -LiteralPath (Join-Path $deps 'transformers'))) {
    throw 'GPU Python or project-local Transformers dependencies are missing; see REPORT.md setup.'
}
function Invoke-NativeCase([string]$device, [string]$label, [int]$threadCount = 8) {
    & (Join-Path $root 'run_torch_npu.ps1') -Script $runner -ScriptArguments '--device',$device,'--label',$label,'--threads',"$threadCount"
}
Invoke-NativeCase 'prepare' 'fixture'
foreach ($threadCount in 1,8) { Invoke-NativeCase 'cpu' "cpu-native-$threadCount" $threadCount }
$oldPath = $env:PATH
$oldPythonPath = $env:PYTHONPATH
try {
    $prefix = Split-Path $XpuPython -Parent
    $env:PATH = "$prefix;$prefix\Library\bin;$prefix\Scripts;$oldPath"
    $env:PYTHONPATH = "$deps;$root"
    foreach ($threadCount in 1,8) {
        & $XpuPython $runner --device cpu --label "cpu-xpu-$threadCount" --threads $threadCount
        if ($LASTEXITCODE -ne 0) { throw "CPU comparison failed: $LASTEXITCODE" }
    }
    & $XpuPython $runner --device xpu --label xpu --threads 8
    if ($LASTEXITCODE -ne 0) { throw "GPU comparison failed: $LASTEXITCODE" }
} finally {
    $env:PATH = $oldPath
    $env:PYTHONPATH = $oldPythonPath
}
Invoke-NativeCase 'npu' 'npu'
Invoke-NativeCase 'audit' 'audit'
