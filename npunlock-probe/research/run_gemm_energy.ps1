param([ValidateRange(10,60)][int]$Seconds=25)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location -LiteralPath $root
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$env:NPUNLOCK_GRAPH_CACHE=$null
$out=Join-Path $PSScriptRoot 'results\gemm-energy'
$runner=Join-Path $root 'torch_npu_gemm_energy.py'
$calibration=Get-Content -LiteralPath (Join-Path $out 'calibration-complete.json') -Raw | ConvertFrom-Json
if ($calibration.status -ne 'passed') { throw 'DPU calibration did not pass' }
$fp16=$calibration.rows | Where-Object { $_.status -eq 'passed' -and $_.meta.m -eq 1024 -and $_.meta.groups -eq 4 -and $_.meta.precision -eq 'float16' } | Sort-Object teraops -Descending | Select-Object -First 1
$int8=$calibration.rows | Where-Object { $_.status -eq 'passed' -and $_.meta.m -eq 1024 -and $_.meta.groups -eq 4 -and $_.meta.precision -eq 'int8' } | Sort-Object teraops -Descending | Select-Object -First 1
if (-not $fp16 -or -not $int8) { throw 'No validated four-matrix DPU configuration' }
$plan=@{status='running';seconds=$Seconds;fp16=$fp16.meta;int8=$int8.meta;cases=@();idle=@()}
function Save-Plan {
    $plan | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $out 'measurement-plan.json') -Encoding UTF8
}
function Idle-Window([string]$name) {
    $begin=([DateTime]::UtcNow.Ticks-621355968000000000)*100
    Start-Sleep -Seconds 10
    $end=([DateTime]::UtcNow.Ticks-621355968000000000)*100
    $plan.idle += @{name=$name;start_unix_ns=$begin;end_unix_ns=$end}
    Save-Plan
}
try {
    Idle-Window 'before'
    $order=@('cpu-f16','cpu-f32','cpu-native-f32','gpu-f16','npu-f16','npu-int8')
    foreach ($round in 1,2) {
        $current=@($order)
        if ($round -eq 2) { [Array]::Reverse($current) }
        foreach ($case in $current) {
            $label="r$round-$case"
            Write-Output "CASE START $label"
            $device=if ($case -like 'npu-*') {'npu'} elseif ($case -like 'gpu-*') {'xpu'} else {'cpu'}
            $precision=if ($case -like '*int8') {'int8'} elseif ($case -like '*f16') {'float16'} else {'float32'}
            $layout='chain4'; $profile='default'
            if ($device -eq 'npu') {
                $selected=if ($precision -eq 'int8') {$int8} else {$fp16}
                $layout=$selected.meta.layout; $profile=$selected.meta.profile
            }
            $argv=@('--device',$device,'--precision',$precision,'--layout',$layout,'--profile',$profile,'--seconds',"$Seconds",'--label',$label)
            if ($device -eq 'npu' -or $case -eq 'cpu-native-f32') {
                & (Join-Path $root 'run_torch_npu.ps1') -Script $runner -ScriptArguments $argv
            } else {
                $python='C:\Users\LC\miniconda3\envs\blood\python.exe'
                $prefix=Split-Path -Parent $python
                $oldPath=$env:PATH
                try {
                    $env:PATH="$prefix;$prefix\Library\bin;$prefix\Scripts;$oldPath"
                    & $python -B $runner @argv
                    if ($LASTEXITCODE -ne 0) { throw "Host case failed: $label" }
                } finally { $env:PATH=$oldPath }
            }
            $result=Get-Content -LiteralPath (Join-Path $out "$label.json") -Raw | ConvertFrom-Json
            if ($result.status -ne 'passed') { throw "Case failed: $label" }
            $plan.cases += $label
            Save-Plan
            Write-Output "CASE DONE $label"
        }
    }
    Idle-Window 'after'
    $plan.status='passed'
} catch {
    $plan.status='failed'; $plan.error=$_ | Out-String
    throw
} finally { Save-Plan }
