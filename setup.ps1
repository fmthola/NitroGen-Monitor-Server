# NitroGen Monitor & Server - Setup Script
# ==========================================
# This script validates your environment and sets up everything needed to run NitroGen

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  NitroGen Monitor & Server Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$errors = @()

# Check Python
Write-Host "[1/5] Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -ge 3 -and $minor -ge 10) {
            Write-Host "  OK: $pythonVersion" -ForegroundColor Green
        } else {
            $errors += "Python 3.10+ required, found $pythonVersion"
            Write-Host "  FAIL: Need Python 3.10+, found $pythonVersion" -ForegroundColor Red
        }
    }
} catch {
    $errors += "Python not found"
    Write-Host "  FAIL: Python not installed" -ForegroundColor Red
}

# Check NVIDIA GPU
Write-Host "[2/5] Checking NVIDIA GPU..." -ForegroundColor Yellow
try {
    $nvidiaSmi = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK: $nvidiaSmi" -ForegroundColor Green
    } else {
        $errors += "NVIDIA GPU not detected"
        Write-Host "  FAIL: No NVIDIA GPU found" -ForegroundColor Red
    }
} catch {
    $errors += "nvidia-smi not found"
    Write-Host "  FAIL: NVIDIA drivers not installed" -ForegroundColor Red
}

# Check ViGEmBus
Write-Host "[3/5] Checking ViGEmBus Driver..." -ForegroundColor Yellow
$vigem = Get-PnpDevice -FriendlyName "*ViGEm*" -ErrorAction SilentlyContinue
if ($vigem) {
    Write-Host "  OK: ViGEmBus driver installed" -ForegroundColor Green
} else {
    $errors += "ViGEmBus not installed"
    Write-Host "  FAIL: ViGEmBus not found" -ForegroundColor Red
    Write-Host "  Download from: https://github.com/nefarius/ViGEmBus/releases" -ForegroundColor Yellow
}

# Check/Create virtual environment
Write-Host "[4/5] Setting up Python environment..." -ForegroundColor Yellow
if (-not (Test-Path ".\venv")) {
    Write-Host "  Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}
Write-Host "  OK: Virtual environment ready" -ForegroundColor Green

# Install dependencies
Write-Host "[5/5] Installing dependencies..." -ForegroundColor Yellow
.\venv\Scripts\activate
pip install --upgrade pip -q
pip install -e ".[serve,play]" -q
pip install huggingface_hub -q
Write-Host "  OK: Dependencies installed" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

if ($errors.Count -eq 0) {
    Write-Host "  Setup Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Download the model (if not already done):"
    Write-Host "   huggingface-cli download nvidia/NitroGen ng.pt --local-dir ./models"
    Write-Host ""
    Write-Host "2. Run the system:"
    Write-Host "   .\run.ps1 -GameProcess 'YourGame.exe'"
} else {
    Write-Host "  Setup Failed - Fix these issues:" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Cyan
    foreach ($err in $errors) {
        Write-Host "  - $err" -ForegroundColor Red
    }
}
