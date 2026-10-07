param([ValidateRange(1,65535)][int]$Port=8765, [ValidateRange(1,65535)][int]$ModelPort=8082)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
$python=Join-Path $projectRoot 'venv\Scripts\python.exe'
$manifest=Join-Path $projectRoot 'tools\graphviz\manifest.json'
if (-not (Test-Path -LiteralPath $manifest)) { throw 'Run .\venv\Scripts\python.exe -B scripts/setup_graphviz.py first.' }
$ledger=Join-Path $projectRoot 'data\training\budget.json'
if ((Test-Path -LiteralPath $ledger) -and (Get-Content -LiteralPath $ledger -Raw | ConvertFrom-Json).active) { throw 'Wait for the active training/evaluation job to finish.' }
$manualLock=Join-Path $projectRoot 'data\runtime\manual.lock'
if (Test-Path -LiteralPath $manualLock) {
    $owner=Get-Content -LiteralPath $manualLock -Raw | ConvertFrom-Json
    if (Get-Process -Id $owner.pid -ErrorAction SilentlyContinue) { throw 'Wait for the active case to finish before restarting.' }
}
$selectedFile=Join-Path $projectRoot 'data\models\selected_model.json'
$choice=if (Test-Path -LiteralPath $selectedFile) { (Get-Content -LiteralPath $selectedFile -Raw | ConvertFrom-Json).selected } else { 'base' }
if ($choice -eq 'd4') {
    $adapter=(Resolve-Path -LiteralPath (Join-Path $projectRoot 'data\models\d4_best.gguf')).Path
    $loaded=@()
    try { $loaded=@(Invoke-RestMethod -Uri "http://127.0.0.1:$ModelPort/lora-adapters" -NoProxy -TimeoutSec 2) } catch { }
    if ($loaded.Count -ne 1 -or $loaded[0].path -ne $adapter) { & (Join-Path $PSScriptRoot 'stop_local_model.ps1') }
    & (Join-Path $PSScriptRoot 'start_local_model.ps1') -ModelSize 4B -Port $ModelPort -AdapterPath $adapter
    Invoke-RestMethod -Uri "http://127.0.0.1:$ModelPort/lora-adapters" -Method Post -ContentType 'application/json' -Body '[{"id":0,"scale":1.0}]' -NoProxy | Out-Null
} else {
    & (Join-Path $PSScriptRoot 'start_local_model.ps1') -ModelSize 4B -Port $ModelPort
    $loaded=@(Invoke-RestMethod -Uri "http://127.0.0.1:$ModelPort/lora-adapters" -NoProxy)
    if ($loaded.Count) { Invoke-RestMethod -Uri "http://127.0.0.1:$ModelPort/lora-adapters" -Method Post -ContentType 'application/json' -Body '[{"id":0,"scale":0.0}]' -NoProxy | Out-Null }
}
$logDir=Join-Path $projectRoot 'data\runtime'
$pidFile=Join-Path $logDir 'demo.pid'
if (Test-Path -LiteralPath $pidFile) {
    $existing=Get-Process -Id ([int](Get-Content -LiteralPath $pidFile)) -ErrorAction SilentlyContinue
    if ($existing -and $existing.Path -eq $python) {
        Write-Output "Input form is already running. Open http://127.0.0.1:$Port"
        return
    }
}
$script=Join-Path $PSScriptRoot 'serve_demo.py'
$demoArgs=@('-u','-B',('"'+$script+'"'),'--port',$Port,'--model-port',$ModelPort)
$process=Start-Process -FilePath $python -ArgumentList $demoArgs -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'demo.stdout.log') -RedirectStandardError (Join-Path $logDir 'demo.stderr.log')
$process.Id | Set-Content -LiteralPath $pidFile
for ($attempt=0;$attempt -lt 40;$attempt++) {
    $process.Refresh()
    if ($process.HasExited) { throw 'Input form failed to start. Inspect data/runtime/demo.stderr.log.' }
    try {
        $state=Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/state" -NoProxy -TimeoutSec 2
        if ($state.token) { Write-Output "Open the clinical input form: http://127.0.0.1:$Port"; return }
    } catch { }
    Start-Sleep -Milliseconds 250
}
throw 'Input form did not become ready. Inspect data/runtime/demo.stderr.log.'
