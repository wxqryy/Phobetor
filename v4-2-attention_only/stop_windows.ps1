$stop = Join-Path $PSScriptRoot 'STOP'
New-Item -ItemType File -Path $stop -Force | Out-Null
Write-Output "Graceful stop requested. The current update will finish, then an emergency checkpoint will be saved to $PSScriptRoot\checkpoints_cuda."
