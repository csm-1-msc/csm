# Cat Monitor - Windows PowerShell Startup Script
#
# Usage:
#   .\start_cat_monitor.ps1              # Run in foreground (with display)
#   .\start_cat_monitor.ps1 -NoDisplay   # Run in background (no display)
#   .\start_cat_monitor.ps1 -Help        # Show help information

param(
    [switch]$NoDisplay,
    [switch]$Help,
    [string]$Camera = "0",
    [string]$Model = "models/cat_furniture_detector.pt",
    [string]$Storage = "./storage",
    [float]$Confidence = 0.5
)

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = $ScriptDir

# Show help
if ($Help) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Cat Monitor - Startup Help" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\start_cat_monitor.ps1 [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -NoDisplay    No display mode (background, for servers)" -ForegroundColor White
    Write-Host "  -Camera <ID>  Camera device index (default: 0)" -ForegroundColor White
    Write-Host "  -Model <path> YOLO model path" -ForegroundColor White
    Write-Host "  -Storage <path> Storage root directory" -ForegroundColor White
    Write-Host "  -Confidence   Detection confidence threshold (default: 0.5)" -ForegroundColor White
    Write-Host "  -Help         Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\start_cat_monitor.ps1                           # Foreground mode" -ForegroundColor White
    Write-Host "  .\start_cat_monitor.ps1 -NoDisplay                # Background mode" -ForegroundColor White
    Write-Host "  .\start_cat_monitor.ps1 -Camera 1                 # Use camera 1" -ForegroundColor White
    Write-Host "  .\start_cat_monitor.ps1 -Camera 0 -NoDisplay      # Camera 0, no display" -ForegroundColor White
    Write-Host ""
    exit 0
}

# Switch to project root
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Cat Monitor - Starting..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Set-Location -Path $ProjectRoot
Write-Host "[OK] Changed to project directory: $ProjectRoot" -ForegroundColor Cyan

# Set PYTHONPATH environment variable
$env:PYTHONPATH = $ProjectRoot
Write-Host "[OK] PYTHONPATH set to: $env:PYTHONPATH" -ForegroundColor Cyan

# Check Python availability
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python not found. Please install Python 3.8+" -ForegroundColor Red
    Write-Host ""
    Write-Host "Download: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] Python version: $($pythonCmd.Version.ToString())" -ForegroundColor Cyan

# Build run arguments
$runArgs = @()
if ($NoDisplay) {
    $runArgs += "--no-display"
    Write-Host "[INFO] Mode: Background (no display)" -ForegroundColor Yellow
} else {
    Write-Host "[INFO] Mode: Foreground (with display)" -ForegroundColor Green
}

if ($Camera -ne "0") {
    $runArgs += "-c"
    $runArgs += $Camera
    Write-Host "[INFO] Camera source: $Camera" -ForegroundColor Yellow
}

if ($Model -ne "models/cat_furniture_detector.pt") {
    $runArgs += "-m"
    $runArgs += $Model
    Write-Host "[INFO] Model path: $Model" -ForegroundColor Yellow
}

if ($Storage -ne "./storage") {
    $runArgs += "-s"
    $runArgs += $Storage
    Write-Host "[INFO] Storage directory: $Storage" -ForegroundColor Yellow
}

if ($Confidence -ne 0.5) {
    $runArgs += "--confidence"
    $runArgs += $Confidence.ToString().Replace(',', '.')
    Write-Host "[INFO] Confidence threshold: $Confidence" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Command:" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "python -m cat_monitor.main $($runArgs -join ' ')" -ForegroundColor White
Write-Host ""

# Start the program
try {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Program running..." -ForegroundColor Green
    Write-Host "  Press Ctrl+C or Q to exit" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    
    python -m cat_monitor.main @runArgs
}
catch {
    Write-Host ""
    Write-Host "[ERROR] Program exited with error: $_" -ForegroundColor Red
    exit 1
}
finally {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Program exited" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
}