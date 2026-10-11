param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if ($root -ne 'E:\Pytorch_xfire_NGPU') { throw 'Unexpected workspace; review cleanup roots first' }
$keptExtensions = @('.md','.py','.ps1','.c','.cpp','.cc','.h','.hpp','.s','.asm','.sh','.cmake','.patch','.diff')
$targets = [Collections.Generic.List[IO.FileInfo]]::new()
function Assert-SafeFile([IO.FileInfo]$file, [string]$boundary) {
    $boundary = [IO.Path]::GetFullPath($boundary).TrimEnd('\')
    $path = [IO.Path]::GetFullPath($file.FullName)
    if (-not $path.StartsWith($boundary+'\',[StringComparison]::OrdinalIgnoreCase)) { throw "Outside cleanup boundary: $path" }
    $cursor = $file
    while ($null -ne $cursor -and $cursor.FullName.Length -ge $boundary.Length) {
        if ($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point: $($cursor.FullName)" }
        $cursor = if ($cursor -is [IO.FileInfo]) { $cursor.Directory } else { $cursor.Parent }
    }
}
function Add-Outputs([string]$directory) {
    if (-not (Test-Path -LiteralPath $directory)) { return }
    foreach ($file in Get-ChildItem -LiteralPath $directory -File -Recurse -Force) {
        if ($file.Extension.ToLower() -in $keptExtensions -or $file.Name -eq 'gpu-model-requirements.txt') { continue }
        Assert-SafeFile $file $directory
        $targets.Add($file)
    }
}
Add-Outputs (Join-Path $root 'npunlock-probe\research\results')
Add-Outputs (Join-Path $root 'npunlock-probe\results')
foreach ($directory in Get-ChildItem -LiteralPath $PSScriptRoot -Directory -Filter '*-before') { Add-Outputs $directory.FullName }
foreach ($file in Get-ChildItem -LiteralPath $PSScriptRoot -File) {
    if ($file.Extension -in '.log','.jsonl' -or ($file.Name -like 'workspace-*.json' -and $file.Name -ne 'workspace-space-20261010-before.json')) {
        Assert-SafeFile $file $PSScriptRoot;$targets.Add($file)
    }
}
$caches = @('C:\Users\LC\AppData\Local\torch-intel-npu\smollm-graph-cache-20261010',
            'C:\Users\LC\AppData\Local\torch-intel-npu\smollm-graph-cache')
foreach ($cache in $caches) {
    if (Test-Path -LiteralPath $cache) {
        foreach ($file in Get-ChildItem -LiteralPath $cache -File -Recurse -Force) {
            Assert-SafeFile $file $cache;$targets.Add($file)
        }
    }
}
$trace = 'C:\Users\LC\AppData\Local\Temp\npunlock_worker_npu_level_zero_umd.etl'
if (Test-Path -LiteralPath $trace) { $file=Get-Item -LiteralPath $trace;Assert-SafeFile $file (Split-Path -Parent $trace);$targets.Add($file) }
foreach ($file in Get-ChildItem -LiteralPath (Join-Path $root 'npunlock-probe\downloads') -File) {
    if ($file.Extension -in '.exe','.zip','.whl') { Assert-SafeFile $file $root;$targets.Add($file) }
}
$targets = @($targets | Sort-Object FullName -Unique)
$report = [ordered]@{mode=$(if ($Apply) {'apply'} else {'preview'});files=$targets.Count;logical_bytes=($targets | Measure-Object Length -Sum).Sum;
    categories='experiment outputs, historical binary snapshots, logs, owned NPU cache/trace, downloaded installers';
    preserved='source, reports, current kernels/DLL/PYD/manifests, model inputs, environments, installed build tools';removed=0;failed=@()}
if ($Apply) {
    if (Get-Process -Name npunlock_worker -ErrorAction SilentlyContinue) { throw 'NPU worker still running' }
    foreach ($file in $targets) {
        try { Remove-Item -LiteralPath $file.FullName -Force -ErrorAction Stop;$report.removed++ }
        catch { $report.failed += [ordered]@{path=$file.FullName;error=$_.Exception.Message} }
    }
    # Empty directories only; sources and reports keep their original paths.
    foreach ($boundary in (@((Join-Path $root 'npunlock-probe\research\results'),(Join-Path $root 'npunlock-probe\results'))+$caches)) {
        if (-not (Test-Path -LiteralPath $boundary)) { continue }
        foreach ($directory in Get-ChildItem -LiteralPath $boundary -Directory -Recurse -Force | Sort-Object FullName -Descending) {
            if ($directory.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Unexpected reparse directory' }
            if (-not (Get-ChildItem -LiteralPath $directory.FullName -Force | Select-Object -First 1)) { Remove-Item -LiteralPath $directory.FullName }
        }
        if ($boundary -in $caches -and -not (Get-ChildItem -LiteralPath $boundary -Force | Select-Object -First 1)) { Remove-Item -LiteralPath $boundary }
    }
}
$report | ConvertTo-Json -Depth 4
