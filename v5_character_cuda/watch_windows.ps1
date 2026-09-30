param([int]$Tail = 20)

$log = Join-Path $PSScriptRoot 'train.log'
if (-not (Test-Path $log)) {
    throw "V5 log not found: $log"
}
Get-Content $log -Tail $Tail -Wait -Encoding UTF8
