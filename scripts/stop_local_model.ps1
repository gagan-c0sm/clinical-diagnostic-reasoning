$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$pidFile = Join-Path $projectRoot 'data\runtime\server.pid'
if (-not (Test-Path -LiteralPath $pidFile)) { Write-Output 'No project server PID recorded.'; return }
$serverProcess = Get-Process -Id ([int](Get-Content -LiteralPath $pidFile)) -ErrorAction SilentlyContinue
if (-not $serverProcess) { Write-Output 'Recorded server process has already stopped.'; return }
$expectedPath = [IO.Path]::GetFullPath((Join-Path $projectRoot 'tools\llama.cpp\llama-server.exe'))
if ($serverProcess.Path -ne $expectedPath) { throw 'PID belongs to another executable; refusing to stop it.' }
Stop-Process -Id $serverProcess.Id
Write-Output 'Project model server stopped; GPU memory released.'
