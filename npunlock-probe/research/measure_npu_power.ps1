param(
    [Parameter(Mandatory=$true)][string]$ToolRoot,
    [ValidateRange(1,900)][int]$Seconds=20,
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$Label='capture',
    [string]$OutputRoot,
    [switch]$IncludeGpu,
    [string]$Application,
    [switch]$PrepareOnly
)
$ErrorActionPreference='Stop'
$workspace=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$tool=(Resolve-Path -LiteralPath $ToolRoot).Path
$exe=Join-Path $tool 'bin\ut.exe'
$driver=Join-Path $tool 'bin\collectors\socwatch\socwatchdrv.sys'
foreach ($file in @($exe,$driver)) {
    $signature=Get-AuthenticodeSignature -LiteralPath $file
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'CN=Intel Corporation,') {
        throw "Expected a valid Intel signature: $file"
    }
}
if (-not $OutputRoot) { $OutputRoot=Join-Path $workspace 'npunlock-probe\research\results\sustained-training\power' }
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot)
$boundary=Join-Path $workspace 'npunlock-probe\research\results'
if (-not $OutputRoot.StartsWith($boundary+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Power output must be inside workspace research results' }
$out=Join-Path $OutputRoot $Label
$config=Join-Path $out 'power-config.json'
$arguments=@('--enable','socwatch','-c','-p',$config,'-s','1000','-t',"$Seconds")
$metrics=@('npu-pwr','pkg-pwr')
if ($IncludeGpu) { $metrics += 'vccgt-pwr' }
if ($Application) { $arguments += @('-a',$Application) }
if ($PrepareOnly) {
    [pscustomobject]@{executable=$exe;arguments=$arguments;metrics=$metrics;
        administrator_required=$true;driver=$driver;output=$out} | ConvertTo-Json -Depth 3
    return
}
$principal=[Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Intel UT power collection requires an administrator PowerShell; no collection started.'
}
New-Item -ItemType Directory -Path $out -Force | Out-Null
foreach ($name in @('power-config.json','power-output.txt','power-complete.json')) {
    if (Test-Path -LiteralPath (Join-Path $out $name)) { throw "Existing power result: $name" }
}
$configuration=@{collectorIdentifier=@{name='socwatch';guid='54B46B56-5439-4D29-8B07-15863D1F6EC6';version='0.1.0'};
    collectionParams=@{samplingInterval=1000;isContinuous=$true;maxDetail=$true};
    metrics=$metrics;outputParams=@{outputFormats=@('uci-bin','uci-text')}} |
    ConvertTo-Json -Depth 5
[IO.File]::WriteAllText($config,$configuration,[Text.UTF8Encoding]::new($false))
$oldPath=$env:PATH
$started=[DateTimeOffset]::UtcNow.ToString('o')
try {
    $env:PATH="$(Join-Path $tool 'bin');$(Join-Path $tool 'bin\collectors\tracers');$oldPath"
    $ErrorActionPreference='Continue'
    & $exe @arguments 2>&1 | Tee-Object -FilePath (Join-Path $out 'power-output.txt')
    $code=$LASTEXITCODE
    $ErrorActionPreference='Stop'
    @{exit_code=$code;started_utc=$started;ended_utc=[DateTimeOffset]::UtcNow.ToString('o');seconds=$Seconds} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'power-complete.json') -Encoding UTF8
    if ($code -ne 0) { throw "Intel UT failed: $code; inspect power-output.txt" }
} finally {
    $env:PATH=$oldPath
}
