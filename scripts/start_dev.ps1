# Start the FabriCare API. Use -WiFi to skip USB port forwarding.
param([switch]$Ensemble, [switch]$WiFi)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv-api\Scripts\python.exe'
$api = Join-Path $root 'fabricare_api.py'

if (-not (Test-Path -LiteralPath $python)) {
  throw 'Run .\scripts\setup_api.ps1 first to create the Python 3.12 API environment.'
}

if (-not $WiFi) {
  $adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
  if (-not (Test-Path -LiteralPath $adb)) {
    $adbCommand = Get-Command adb -ErrorAction SilentlyContinue
    if (-not $adbCommand) { throw 'ADB not found. Install Android SDK platform-tools or use -WiFi.' }
    $adb = $adbCommand.Source
  }

  Write-Host 'Setting up adb reverse tcp:5000 -> PC:5000 ...'
  & $adb reverse tcp:5000 tcp:5000
  if ($LASTEXITCODE -ne 0) {
    throw 'USB forwarding failed. Check adb devices, or use -WiFi with a LAN-configured Android build.'
  }
  & $adb reverse --list
}

Push-Location $root
try {
  @'
from db_config import DbConfig
from database import ping_database
try:
    ping_database()
except Exception as error:
    code = getattr(error, "errno", "unknown")
    print(f"MySQL check failed (code {code}) at {DbConfig.host}:{DbConfig.port}.")
    print("Check MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD and MYSQL_DATABASE in the local .env.")
    print("Code 2003: server/port unreachable. Code 1045: credentials rejected. Code 1049: database missing; start fabricare_api.py once to initialize it.")
    raise SystemExit(1)
print("MySQL connection verified.")
'@ | & $python -
  if ($LASTEXITCODE -ne 0) { throw 'Resolve the MySQL connection error above before starting the app.' }

Write-Host 'Starting FabriCare API on port 5000 ...'
if ($WiFi) {
  Write-Host 'Connect both devices to the same Wi-Fi. The Android build must use this PC''s Wi-Fi address on port 5000.'
}
$previousClassifier = $env:FABRIC_CLASSIFIER
try {
  if ($Ensemble) { $env:FABRIC_CLASSIFIER = 'ensemble' }
  & $python $api
} finally {
  $env:FABRIC_CLASSIFIER = $previousClassifier
}
} finally {
  Pop-Location
}
