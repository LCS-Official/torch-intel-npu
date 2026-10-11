param([string]$Python = 'python', [string]$Toolchain)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$Python = (Get-Command $Python -CommandType Application -ErrorAction Stop).Source
$prefix = & $Python -c 'import sys; print(sys.prefix)'
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the selected Python environment' }
if (-not $Toolchain -and (Test-Path -LiteralPath "$root\npunlock-probe\toolchain\msvc")) {
    $Toolchain = "$root\npunlock-probe\toolchain\msvc"
}
. (Join-Path $root 'tools\msvc_environment.ps1')
$saved = @{}
foreach ($name in @('PATH','INCLUDE','LIB')) { $saved[$name] = [Environment]::GetEnvironmentVariable($name) }
try {
    $env:PATH = "$prefix;$prefix\Library\bin;$prefix\Scripts;$env:PATH"
    $tool = Enter-NpuMsvcEnvironment $Toolchain
    $cmake = (Get-Command cmake.exe -ErrorAction Stop).Source
    $build = Join-Path $root 'npunlock\build\source-runtime'
    $runtime = Join-Path $root 'npunlock-probe\native-runtime'
    & $cmake -S "$root\npunlock" -B $build -G Ninja "-DCMAKE_C_COMPILER=$($tool.Compiler)" -DCMAKE_BUILD_TYPE=Release -DNPUNLOCK_BUILD_TESTS=ON -DBUILD_TESTING=ON "-DPython3_EXECUTABLE=$Python"
    if ($LASTEXITCODE -ne 0) { throw 'npunlock CMake configuration failed' }
    & $cmake --build $build --parallel 2 --target npunlock_native npunlock_worker npunlock_test_session_failure npunlock_test_session_chain
    if ($LASTEXITCODE -ne 0) { throw 'npunlock source build failed' }
    foreach ($test in @('npunlock_test_session_failure.exe','npunlock_test_session_chain.exe')) {
        & (Join-Path $build $test)
        if ($LASTEXITCODE -ne 0) { throw "Native regression failed: $test" }
    }
    & $cmake --install $build --prefix $runtime --component PythonRuntime
    if ($LASTEXITCODE -ne 0) { throw 'npunlock native runtime installation failed' }
    $session = Join-Path $root 'npunlock-probe\session-runtime'
    New-Item -ItemType Directory -Force -Path $session | Out-Null
    # The rebuilt full library exports the session API; no separate MinGW is needed.
    Copy-Item -LiteralPath "$runtime\bin\npunlock.dll" -Destination "$session\npunlock_session.dll"
    $provenance = @{
        mode = 'MSVC source-built full npunlock library also provides session API'
        compiler = $tool.Compiler; windows_sdk = $tool.WindowsSdk
        dllSha256 = (Get-FileHash "$session\npunlock_session.dll").Hash.ToLower()
        workerSha256 = (Get-FileHash "$runtime\bin\npunlock_worker.exe").Hash.ToLower()
        sources = @(Get-ChildItem -LiteralPath "$root\npunlock\src" -Recurse -File | Where-Object Extension -In '.c','.h' | ForEach-Object {
            @{ path = $_.FullName.Substring($root.Length + 1); sha256 = (Get-FileHash $_.FullName).Hash.ToLower() }
        })
    }
    $provenance | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$session\build.json" -Encoding utf8
    $provenance | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$runtime\build.json" -Encoding utf8
    & "$root\build_torch_npu_device.ps1" -Python $Python -Toolchain $Toolchain
    if ($LASTEXITCODE -ne 0) { throw 'Torch extension build failed' }
    Write-Output 'Source build and native session regressions passed.'
} finally {
    foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name]) }
}
