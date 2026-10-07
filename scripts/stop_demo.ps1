$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
$python=Join-Path $projectRoot 'venv\Scripts\python.exe'
$pidFile=Join-Path $projectRoot 'data\runtime\demo.pid'
if (Test-Path -LiteralPath $pidFile) {
    $process=Get-Process -Id ([int](Get-Content -LiteralPath $pidFile)) -ErrorAction SilentlyContinue
    if ($process) {
        $details=Get-CimInstance Win32_Process -Filter "ProcessId=$($process.Id)"
        $script=Join-Path $projectRoot 'scripts\serve_demo.py'
        if ($process.Path -ne $python -or -not $details.CommandLine.Contains($script)) { throw 'PID does not belong to the project input form.' }
        Stop-Process -Id $process.Id
    }
    Remove-Item -LiteralPath $pidFile -Force
}
& (Join-Path $PSScriptRoot 'stop_local_model.ps1')
Write-Output 'Input form stopped.'
