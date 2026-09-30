$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = 'D:\PhobetorBench\.venv\Scripts\python.exe'
$name = 'PhobetorV5Training'
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
if (-not (Test-Path $python)) {
    throw "Python not found: $python"
}
$existing = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') {
    throw 'V5 preparation or training is already running'
}
$stop = Join-Path $root 'STOP'
if (Test-Path $stop) {
    Remove-Item $stop
}
$action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument '/d /c "run_windows.cmd >> train.log 2>> train.err.log"' -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = 'PT1M'
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $name
Get-ScheduledTask -TaskName $name | Select-Object TaskName,State
