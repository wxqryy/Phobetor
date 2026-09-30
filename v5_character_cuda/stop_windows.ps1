$stop = Join-Path $PSScriptRoot 'STOP'
New-Item -ItemType File -Path $stop -Force | Out-Null
Write-Output "Graceful training stop requested: $stop"
