$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = 'D:\PhobetorBench\.venv\Scripts\python.exe'
$name = 'PhobetorV4-2Training'
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
if (-not (Test-Path $python)) {
    throw "Python not found: $python"
}
$existing = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') {
    throw 'V4-2 training is already running'
}
$stop = Join-Path $root 'STOP'
if (Test-Path $stop) {
    Remove-Item $stop
}
$command = "cd /d $root && $python -u train.py --micro-batch 16 > train.log 2> train.err.log"
$action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/d /c `"$command`"" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddHours(1)
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $name
Get-ScheduledTask -TaskName $name | Select-Object TaskName,State
