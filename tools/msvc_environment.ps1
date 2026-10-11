function Enter-NpuMsvcEnvironment([string]$Toolchain) {
    if ($Toolchain) {
        $Toolchain = (Resolve-Path -LiteralPath $Toolchain).Path
        $vc = Get-ChildItem -LiteralPath (Join-Path $Toolchain 'VC\Tools\MSVC') -Directory | Sort-Object Name -Descending | Select-Object -First 1
        $sdk = Get-ChildItem -LiteralPath (Join-Path $Toolchain 'Windows Kits\10\Include') -Directory | Sort-Object Name -Descending | Select-Object -First 1
        $compiler = Join-Path $vc.FullName 'bin\Hostx64\x64\cl.exe'
        if (-not (Test-Path -LiteralPath $compiler)) { throw "MSVC x64 compiler not found: $compiler" }
        $env:PATH = "$(Split-Path $compiler);$Toolchain\Windows Kits\10\bin\$($sdk.Name)\x64;$env:PATH"
        $env:INCLUDE = "$($vc.FullName)\include;$Toolchain\Windows Kits\10\Include\$($sdk.Name)\ucrt;$Toolchain\Windows Kits\10\Include\$($sdk.Name)\shared;$Toolchain\Windows Kits\10\Include\$($sdk.Name)\um"
        $env:LIB = "$($vc.FullName)\lib\x64;$Toolchain\Windows Kits\10\Lib\$($sdk.Name)\ucrt\x64;$Toolchain\Windows Kits\10\Lib\$($sdk.Name)\um\x64"
        return @{ Compiler = $compiler; WindowsSdk = $sdk.Name }
    }
    $compiler = Get-Command cl.exe -ErrorAction SilentlyContinue
    if (-not $compiler -or -not $env:INCLUDE -or -not $env:LIB) {
        throw 'Use an x64 Visual Studio Developer PowerShell, or pass -Toolchain with VC and Windows Kits directories'
    }
    return @{ Compiler = $compiler.Source; WindowsSdk = $env:WindowsSDKVersion }
}
