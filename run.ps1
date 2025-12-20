# NitroGen Monitor & Server - Run Script
# ========================================
# Usage: .\run.ps1 -GameProcess "YourGame.exe"

param(
    [Parameter(Mandatory=$true)]
    [string]$GameProcess,
    
    [int]$Timesteps = 2,
    [int]$ServerPort = 5555,
    [int]$MonitorPort = 5556,
    [switch]$NoMonitor
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  NitroGen Gaming AI" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Game Process: $GameProcess" -ForegroundColor Yellow
Write-Host "Timesteps: $Timesteps (lower = faster)" -ForegroundColor Yellow
Write-Host ""

# Check if game is running
$game = Get-Process -Name ($GameProcess -replace '\.exe$','') -ErrorAction SilentlyContinue
if (-not $game) {
    Write-Host "ERROR: Game not running!" -ForegroundColor Red
    Write-Host "Please start $GameProcess first, then run this script." -ForegroundColor Yellow
    exit 1
}
Write-Host "OK: Found $GameProcess (PID: $($game.Id))" -ForegroundColor Green

# Check if model exists
if (-not (Test-Path ".\models\ng.pt")) {
    Write-Host "ERROR: Model not found!" -ForegroundColor Red
    Write-Host "Download it with:" -ForegroundColor Yellow
    Write-Host "  huggingface-cli download nvidia/NitroGen ng.pt --local-dir ./models" -ForegroundColor White
    exit 1
}
Write-Host "OK: Model found" -ForegroundColor Green
Write-Host ""

# Activate environment
.\venv\Scripts\activate

Write-Host "Starting NitroGen components..." -ForegroundColor Yellow
Write-Host "(Press Ctrl+C in any window to stop)" -ForegroundColor Gray
Write-Host ""

# Start server in new window
$serverCmd = "cd '$PWD'; .\venv\Scripts\activate; python scripts/serve.py models/ng.pt --port $ServerPort --timesteps $Timesteps"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $serverCmd

Write-Host "Started: Model Server (port $ServerPort)" -ForegroundColor Green

# Wait for server to load
Write-Host "Waiting for model to load (30 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Start monitor in new window (unless disabled)
if (-not $NoMonitor) {
    $monitorCmd = "cd '$PWD'; .\venv\Scripts\activate; python scripts/monitor.py --port $MonitorPort"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $monitorCmd
    Write-Host "Started: Monitor Dashboard (port $MonitorPort)" -ForegroundColor Green
}

# Start player in new window
$playerCmd = "cd '$PWD'; .\venv\Scripts\activate; python scripts/play_simple.py --process '$GameProcess' --fps 60 --monitor-port $MonitorPort"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $playerCmd

Write-Host "Started: Game Agent" -ForegroundColor Green
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AI is now controlling $GameProcess!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To stop: Close the PowerShell windows or press Ctrl+C in each" -ForegroundColor Yellow
