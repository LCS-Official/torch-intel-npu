param([switch]$Quick, [switch]$Session, [switch]$Device, [switch]$Training, [switch]$Models, [switch]$Mixed,
      [switch]$Large, [ValidateSet('all','pointwise','training','conv')][string]$LargeCase='all',
      [switch]$Stream, [ValidateSet('all','k-f16','k-f32','conv-f16','conv-f32','conv-geometry','conv-dense')][string]$StreamCase='all',
      [switch]$Benchmark, [ValidateSet('all','float16','float32')][string]$BenchmarkPrecision='all',
      [ValidateSet('all','elementwise_1m','mm_64','mm_256','linear_forward','linear_train_large_batch','conv_forward','conv_train','resnet_train')][string]$BenchmarkWorkload='all',
      [string]$XpuPython, [switch]$BenchmarkResume, [string]$BenchmarkOutputDirectory, [string]$Script,
      [string[]]$ScriptArguments = @(), [string]$Python, [string]$NativeDirectory)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
if (-not $Python) {
    $Python = Join-Path $root '.conda\npunlock\python.exe'
    if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }
}
$Python = (Get-Command $Python -CommandType Application -ErrorAction Stop).Source
$prefix = & $Python -c 'import sys; print(sys.prefix)'
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the selected Python environment' }
if (-not $NativeDirectory) {
    $NativeDirectory = Join-Path $root 'npunlock-probe\native-runtime\bin'
    if (-not (Test-Path -LiteralPath (Join-Path $NativeDirectory 'npunlock.dll'))) {
        $NativeDirectory = Join-Path $prefix 'Lib\site-packages\npunlock\_bin'
    }
}
$oldPath = $env:PATH
$oldPythonPath = $env:PYTHONPATH
$oldNativeDirectory = $env:NPUNLOCK_NATIVE_DIR
try {
    $env:PATH = "$prefix;$prefix\Library\bin;$prefix\Scripts;$oldPath"
    $env:PYTHONPATH = "$root;$(Join-Path $root 'npunlock\python');$oldPythonPath"
    $env:NPUNLOCK_NATIVE_DIR = $NativeDirectory
    $arguments = @((Join-Path $root 'torch_npu_demo.py'))
    if ($Session) { $arguments = @((Join-Path $root 'torch_npu_session_demo.py')) }
    if ($Device) { $arguments = @((Join-Path $root 'torch_npu_device_demo.py')) }
    if ($Training) { $arguments = @((Join-Path $root 'torch_npu_training_demo.py')) }
    if ($Models) { $arguments = @((Join-Path $root 'torch_npu_training_demo.py'), '--models') }
    if ($Mixed) { $arguments = @((Join-Path $root 'torch_npu_mixed_demo.py')) }
    if ($Large) { $arguments = @((Join-Path $root 'torch_npu_large_demo.py'), '--case', $LargeCase) }
    if ($Stream) { $arguments = @((Join-Path $root 'torch_npu_stream_demo.py'), '--case', $StreamCase) }
    if ($Benchmark) {
        $arguments = @((Join-Path $root 'torch_npu_benchmark.py'), '--precision', $BenchmarkPrecision, '--workload', $BenchmarkWorkload)
        if ($XpuPython) { $arguments += @('--xpu-python', (Resolve-Path -LiteralPath $XpuPython).Path) }
        if ($BenchmarkResume) { $arguments += '--resume' }
        if ($BenchmarkOutputDirectory) { $arguments += @('--output-dir', $BenchmarkOutputDirectory) }
    }
    if ($Script) { $arguments = @((Resolve-Path -LiteralPath $Script).Path) + $ScriptArguments }
    elseif ($Quick -and -not ($Large -or $Stream -or $Benchmark)) { $arguments += '--quick' }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Torch NPU validation failed: $LASTEXITCODE" }
} finally {
    $env:PATH = $oldPath
    $env:PYTHONPATH = $oldPythonPath
    $env:NPUNLOCK_NATIVE_DIR = $oldNativeDirectory
}
