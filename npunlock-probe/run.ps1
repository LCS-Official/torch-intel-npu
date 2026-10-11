param(
    [string[]]$Cases = @('abs', 'add', 'subtract', 'multiply', 'matmul'),
    [switch]$Shared
)
$ErrorActionPreference = 'Stop'
$prefix = Join-Path (Split-Path $PSScriptRoot -Parent) '.conda\npunlock'
$previousPath = $env:PATH
$previousMoviTools = $env:NPUNLOCK_MOVITOOLS_DIR
try {
    $env:PATH = "$prefix;$prefix\Library\bin;$prefix\Scripts;$previousPath"
    $env:NPUNLOCK_MOVITOOLS_DIR = Join-Path $PSScriptRoot 'toolchain\Driver\MVC_DEPEND'
    $arguments = @((Join-Path $PSScriptRoot 'probe.py'), '--cases') + $Cases
    if (-not $Shared) { $arguments += '--copied-only' }
    & "$prefix\python.exe" @arguments
    $probeExit = $LASTEXITCODE
} finally {
    $env:PATH = $previousPath
    $env:NPUNLOCK_MOVITOOLS_DIR = $previousMoviTools
}
exit $probeExit
