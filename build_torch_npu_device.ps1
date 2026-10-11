param([string]$Toolchain, [string]$Python)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
if (-not $Python) {
    $Python = Join-Path $root '.conda\npunlock\python.exe'
    if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }
}
$Python = (Get-Command $Python -CommandType Application -ErrorAction Stop).Source
$info = & $Python -c 'import json,sys,sysconfig,struct,torch; from pathlib import Path; print(json.dumps(dict(prefix=sys.prefix,torch=str(Path(torch.__file__).parent),version=torch.__version__,include=sysconfig.get_path("include"),libs=str(Path(sys.base_prefix)/"libs"),python_lib="python%d%d.lib"%sys.version_info[:2],bits=struct.calcsize("P")*8)))'
if ($LASTEXITCODE -ne 0) { throw 'Cannot import Torch in the selected Python environment' }
$info = $info | ConvertFrom-Json
if ($info.bits -ne 64) { throw 'The NPU backend requires 64-bit Python' }
$prefix = $info.prefix
$torch = $info.torch
$output = Join-Path $root 'npunlock-probe\torch-device'
New-Item -ItemType Directory -Force -Path $output | Out-Null
if (-not $Toolchain -and (Test-Path -LiteralPath "$root\npunlock-probe\toolchain\msvc")) { $Toolchain = "$root\npunlock-probe\toolchain\msvc" }
. (Join-Path $root 'tools\msvc_environment.ps1')
$saved = @{}
foreach ($name in @('PATH','INCLUDE','LIB')) { $saved[$name] = [Environment]::GetEnvironmentVariable($name) }
try {
    $tool = Enter-NpuMsvcEnvironment $Toolchain
    $compiler = $tool.Compiler
    $env:PATH = "$prefix;$prefix\Library\bin;$env:PATH"
    $env:LIB = "$torch\lib;$($info.libs);$env:LIB"
    $arguments = @('/nologo','/LD','/MD','/O2','/std:c++20','/EHsc','/W3','/wd4251','/wd4275',
        "/I$torch\include", "/I$torch\include\torch\csrc\api\include", "/I$($info.include)",
        "/I$root\npunlock\include", "/Fo$output\backend.obj", (Join-Path $output 'backend.cpp'),
        '/link','torch_cpu.lib','torch_python.lib','c10.lib',$info.python_lib,
        "/OUT:$output\_torch_intel_npu.pyd", "/IMPLIB:$output\_torch_intel_npu.lib")
    & $compiler @arguments
    if ($LASTEXITCODE -ne 0) { throw "PyTorch device extension build failed: $LASTEXITCODE" }
    $torchVersion = $info.version
    @{ compiler = $compiler; compiler_sha256 = (Get-FileHash $compiler).Hash.ToLower();
       torch_version = $torchVersion; python = $Python; python_library = $info.python_lib; windows_sdk = $tool.WindowsSdk; arguments = $arguments;
       source_sha256 = (Get-FileHash (Join-Path $output 'backend.cpp')).Hash.ToLower();
       extension_sha256 = (Get-FileHash (Join-Path $output '_torch_intel_npu.pyd')).Hash.ToLower();
       session_sha256 = (Get-FileHash (Join-Path $root 'npunlock-probe\session-runtime\npunlock_session.dll')).Hash.ToLower()
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'build.json') -Encoding utf8
} finally {
    foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name]) }
}
