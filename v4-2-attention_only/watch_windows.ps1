param([int]$Tail = 20)

$log = Join-Path $PSScriptRoot 'train.log'
if (-not (Test-Path $log)) {
    throw "Training log not found: $log"
}
Get-Content $log -Tail $Tail -Wait -Encoding UTF8
