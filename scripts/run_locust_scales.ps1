# Run Locust flood tests against docker-compose at api scales 1, 2, and 4.
# Usage (from project root, Docker Desktop running):
#   powershell -ExecutionPolicy Bypass -File scripts\run_locust_scales.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$ResultsDir = Join-Path $Root "locust\results"
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

$Locust = Join-Path $env:LOCALAPPDATA "Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\locust.exe"
if (-not (Test-Path $Locust)) {
  $Locust = "locust"
}

Write-Host "Building stack..."
docker compose up --build -d --scale api=1
Start-Sleep -Seconds 15

foreach ($n in 1, 2, 4) {
  Write-Host "`n=== Scaling api=$n ==="
  docker compose up -d --scale api=$n --no-recreate
  # Give nginx DNS + model load a moment
  Start-Sleep -Seconds 20

  # Health check
  $ok = $false
  for ($i = 0; $i -lt 30; $i++) {
    try {
      $r = Invoke-WebRequest -Uri "http://localhost:8080/status" -UseBasicParsing -TimeoutSec 10
      if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch {
      Start-Sleep -Seconds 2
    }
  }
  if (-not $ok) { throw "API not healthy at scale $n" }

  $prefix = Join-Path $ResultsDir "run_${n}containers"
  Write-Host "Running Locust -> $prefix"
  & $Locust -f (Join-Path $Root "locust\locustfile.py") `
    --host=http://localhost:8080 `
    --headless -u 50 -r 10 -t 60s `
    --csv=$prefix `
    --only-summary
}

Write-Host "`nDone. CSVs in locust\results\"
Get-ChildItem $ResultsDir -Filter "*_stats.csv" | Select-Object Name, Length
