param([string]$Compiler = 'C:\Program Files\mingw64\bin\gcc.exe')
$ErrorActionPreference = 'Stop'
$sourceRoot = Join-Path $PSScriptRoot 'npunlock'
$outputRoot = Join-Path $PSScriptRoot 'npunlock-probe\session-runtime'
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$sources = @('src/common/buffer.c', 'src/common/checked.c', 'src/common/diagnostic.c',
             'src/common/status.c', 'src/graphinfer/graphinfer.c', 'src/graphinfer/session_win32.c')
$arguments = @('-shared', '-O2', '-std=c17', '-Wall', '-Wextra', '-Werror', '-Wpedantic',
               '-static-libgcc', '-DNPUNLOCK_BUILDING', '-DNPUNLOCK_SHARED',
               '-I', (Join-Path $sourceRoot 'include'), '-I', (Join-Path $sourceRoot 'src'),
               '-I', (Join-Path $sourceRoot 'src/common'))
$arguments += $sources | ForEach-Object { Join-Path $sourceRoot $_ }
$arguments += @('-o', (Join-Path $outputRoot 'npunlock_session.dll'))
$oldCompilerPath = $env:Path
try {
    $env:Path = "$(Split-Path -Parent $Compiler);$oldCompilerPath"
    & $Compiler @arguments
    if ($LASTEXITCODE -ne 0) { throw "Session DLL build failed: $LASTEXITCODE" }
    $testArguments = @('-O2', '-std=c17', '-Wall', '-Wextra', '-Werror', '-Wpedantic', '-static-libgcc',
                      '-I', (Join-Path $sourceRoot 'include'), '-I', (Join-Path $sourceRoot 'src'),
                      '-I', (Join-Path $sourceRoot 'src/common'))
    $testArguments += $sources[0..3] | ForEach-Object { Join-Path $sourceRoot $_ }
    $testArguments += @((Join-Path $sourceRoot 'tests/unit/test_session_failure.c'),
                       '-o', (Join-Path $outputRoot 'test_session_failure.exe'))
    & $Compiler @testArguments
    if ($LASTEXITCODE -ne 0) { throw "Session failure test build failed: $LASTEXITCODE" }
    & (Join-Path $outputRoot 'test_session_failure.exe')
    if ($LASTEXITCODE -ne 0) { throw "Session failure test failed: $LASTEXITCODE" }
    $chainArguments = $testArguments | ForEach-Object {
        if ($_ -eq (Join-Path $sourceRoot 'tests/unit/test_session_failure.c')) { Join-Path $sourceRoot 'tests/unit/test_session_chain.c' }
        elseif ($_ -eq (Join-Path $outputRoot 'test_session_failure.exe')) { Join-Path $outputRoot 'test_session_chain.exe' }
        else { $_ }
    }
    & $Compiler @chainArguments
    if ($LASTEXITCODE -ne 0) { throw "Session chain test build failed: $LASTEXITCODE" }
    & (Join-Path $outputRoot 'test_session_chain.exe')
    if ($LASTEXITCODE -ne 0) { throw "Session chain test failed: $LASTEXITCODE" }
} finally {
    $env:Path = $oldCompilerPath
}
Get-FileHash -LiteralPath (Join-Path $outputRoot 'npunlock_session.dll') -Algorithm SHA256
$provenance = [ordered]@{
    compiler = $Compiler
    compilerVersion = (& $Compiler -dumpfullversion)
    dllSha256 = (Get-FileHash -LiteralPath (Join-Path $outputRoot 'npunlock_session.dll') -Algorithm SHA256).Hash
    sources = @($sources | ForEach-Object {
        @{ path = $_; sha256 = (Get-FileHash -LiteralPath (Join-Path $sourceRoot $_) -Algorithm SHA256).Hash }
    })
    headers = @('include/npunlock/graphinfer.h', 'src/workers/level_zero_min.h') | ForEach-Object {
        @{ path = $_; sha256 = (Get-FileHash -LiteralPath (Join-Path $sourceRoot $_) -Algorithm SHA256).Hash }
    }
    arguments = $arguments
}
$provenance | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $outputRoot 'build.json') -Encoding utf8
