param([ValidateRange(512,32768)][int]$ContextSize = 8192, [ValidateRange(1,65535)][int]$Port = 8081, [ValidateSet('4B','8B')][string]$ModelSize = '4B', [string]$AdapterPath)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$runtimeDir = Join-Path $projectRoot 'tools\llama.cpp'
$server = Get-ChildItem -LiteralPath $runtimeDir -Filter 'llama-server.exe' -Recurse | Select-Object -First 1
$modelFile = Join-Path $projectRoot "data\models\Qwen3-$ModelSize-Q4_K_M.gguf"
if (-not $server -or -not (Test-Path -LiteralPath $modelFile)) { throw 'Run scripts/setup_local_runtime.py first.' }
$logDir = Join-Path $projectRoot 'data\runtime'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$pidFile = Join-Path $logDir 'server.pid'
if (Test-Path -LiteralPath $pidFile) {
    $existingProcess = Get-Process -Id ([int](Get-Content -LiteralPath $pidFile)) -ErrorAction SilentlyContinue
    if ($existingProcess -and $existingProcess.Path -eq $server.FullName) {
        Write-Output "Project model server is already running (PID $($existingProcess.Id))."
        return
    }
}
$serverArgs = @('-m', ('"' + $modelFile + '"'), '--host', '127.0.0.1', '--port', $Port,
    '-c', $ContextSize, '-ngl', '99', '-np', '1', '--jinja', '--no-context-shift',
    '--alias', 'local-clinical')
if ($AdapterPath) {
    $resolvedAdapter = (Resolve-Path -LiteralPath $AdapterPath).Path
    $serverArgs += @('--lora', ('"' + $resolvedAdapter + '"'), '--lora-init-without-apply')
}
$process = Start-Process -FilePath $server.FullName -WorkingDirectory $server.DirectoryName -ArgumentList $serverArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'server.stdout.log') -RedirectStandardError (Join-Path $logDir 'server.stderr.log')
$process.Id | Set-Content -LiteralPath (Join-Path $logDir 'server.pid')
$manifestFile = Join-Path $runtimeDir "manifest-$ModelSize.json"
if (Test-Path -LiteralPath $manifestFile) {
    Copy-Item -LiteralPath $manifestFile -Destination (Join-Path $runtimeDir 'manifest.json')
}
for ($attempt = 0; $attempt -lt 100; $attempt++) {
    $process.Refresh()
    if ($process.HasExited) { throw "Model server exited; inspect $logDir\server.stderr.log" }
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 1 -NoProxy
        if ($health.status -eq 'ok') {
            @{model=$ModelSize; context_size=$ContextSize; port=$Port; pid=$process.Id} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logDir 'config.json')
            Write-Output "Local Qwen $ModelSize ready (PID $($process.Id), port $Port). Logs: $logDir"
            return
        }
    } catch { }
    Start-Sleep -Milliseconds 500
}
throw "Model server did not become ready; inspect $logDir\server.stderr.log"
