# AI Meow Butler (Cat Monitor) - Dependencies Installation Script
# For Windows PowerShell
#
# Usage:
#   .\install_dependencies.ps1              # Install all dependencies
#   .\install_dependencies.ps1 -Step 1      # Step 1 only: Upgrade pip
#   .\install_dependencies.ps1 -Step 2      # Step 2 only: Install basic libs
#   .\install_dependencies.ps1 -Step 3      # Step 3 only: Install data libs
#   .\install_dependencies.ps1 -Step 4      # Step 4 only: Install PyTorch
#   .\install_dependencies.ps1 -Step 5      # Step 5 only: Install ultralytics
#   .\install_dependencies.ps1 -Help        # Show help information

param(
    [switch]$Help,
    [int]$Step = 0,
    [switch]$SkipConfirm
)

# Tsinghua TUNA mirror
$TUNA_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
$PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# Show help
if ($Help) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Cat Monitor - Installation Help" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\install_dependencies.ps1 [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -Step <N>     Run specific step only (1-5)" -ForegroundColor White
    Write-Host "  -SkipConfirm  Skip confirmation prompts" -ForegroundColor White
    Write-Host "  -Help         Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "Steps:" -ForegroundColor Yellow
    Write-Host "  Step 1: Upgrade pip to latest version" -ForegroundColor White
    Write-Host "  Step 2: Install basic libs (opencv-python, numpy, colorama)" -ForegroundColor White
    Write-Host "  Step 3: Install data analysis libs (scikit-learn, pandas, statsmodels)" -ForegroundColor White
    Write-Host "  Step 4: Install PyTorch CPU version" -ForegroundColor White
    Write-Host "  Step 5: Install ultralytics (YOLOv8)" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\install_dependencies.ps1                    # Install all" -ForegroundColor White
    Write-Host "  .\install_dependencies.ps1 -Step 1            # Upgrade pip only" -ForegroundColor White
    Write-Host "  .\install_dependencies.ps1 -Step 4 -SkipConfirm  # Install PyTorch" -ForegroundColor White
    Write-Host ""
    exit 0
}

# Check Python availability
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python not found. Please install Python 3.8+" -ForegroundColor Red
    Write-Host ""
    Write-Host "Download: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Python version: $($pythonCmd.Version.ToString())" -ForegroundColor Cyan

# Welcome message
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Cat Monitor - Installation Wizard" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "This script will install dependencies using Tsinghua mirror." -ForegroundColor White
Write-Host ""

# Step 1: Upgrade pip
function Install-Step1 {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Step 1: Upgrade pip" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Description: Upgrade pip to latest version" -ForegroundColor Yellow
    Write-Host ""
    
    if (-not $SkipConfirm) {
        $confirm = Read-Host "Continue? (Y/n)"
        if ($confirm -eq 'n' -or $confirm -eq 'N') { return }
    }
    
    Write-Host "Command: python -m pip install --upgrade pip -i $TUNA_INDEX" -ForegroundColor Gray
    python -m pip install --upgrade pip -i $TUNA_INDEX
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] pip upgraded successfully" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] pip upgrade may have failed, continuing..." -ForegroundColor Yellow
    }
    Write-Host ""
}

# Step 2: Basic libs
function Install-Step2 {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Step 2: Install Basic Libraries" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Description: Install opencv-python, numpy, colorama" -ForegroundColor Yellow
    Write-Host "Estimated size: ~50-100 MB" -ForegroundColor Yellow
    Write-Host ""
    
    if (-not $SkipConfirm) {
        $confirm = Read-Host "Continue? (Y/n)"
        if ($confirm -eq 'n' -or $confirm -eq 'N') { return }
    }
    
    Write-Host "Command: pip install opencv-python numpy colorama -i $TUNA_INDEX" -ForegroundColor Gray
    pip install opencv-python numpy colorama -i $TUNA_INDEX
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Basic libraries installed successfully" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Basic libraries installation failed" -ForegroundColor Red
        return $false
    }
    Write-Host ""
    return $true
}

# Step 3: Data analysis libs
function Install-Step3 {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Step 3: Install Data Analysis Libraries" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Description: Install scikit-learn, pandas, statsmodels" -ForegroundColor Yellow
    Write-Host "Estimated size: ~100-200 MB" -ForegroundColor Yellow
    Write-Host ""
    
    if (-not $SkipConfirm) {
        $confirm = Read-Host "Continue? (Y/n)"
        if ($confirm -eq 'n' -or $confirm -eq 'N') { return }
    }
    
    Write-Host "Command: pip install scikit-learn pandas statsmodels -i $TUNA_INDEX" -ForegroundColor Gray
    pip install scikit-learn pandas statsmodels -i $TUNA_INDEX
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Data analysis libraries installed successfully" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Data analysis libraries installation failed" -ForegroundColor Red
        return $false
    }
    Write-Host ""
    return $true
}

# Step 4: PyTorch
function Install-Step4 {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Step 4: Install PyTorch (CPU)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Description: Install torch, torchvision, torchaudio (CPU version)" -ForegroundColor Yellow
    Write-Host "Estimated size: ~200-400 MB" -ForegroundColor Yellow
    Write-Host "Note: Downloading from PyTorch official source" -ForegroundColor Yellow
    Write-Host ""
    
    if (-not $SkipConfirm) {
        $confirm = Read-Host "Continue? (Y/n)"
        if ($confirm -eq 'n' -or $confirm -eq 'N') { return }
    }
    
    Write-Host "Command: pip install torch torchvision torchaudio --index-url $PYTORCH_CPU_INDEX" -ForegroundColor Gray
    pip install torch torchvision torchaudio --index-url $PYTORCH_CPU_INDEX
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] PyTorch installed successfully" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] PyTorch installation failed" -ForegroundColor Red
        return $false
    }
    Write-Host ""
    return $true
}

# Step 5: ultralytics
function Install-Step5 {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Step 5: Install ultralytics (YOLOv8)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Description: Install ultralytics (YOLOv8 object detection)" -ForegroundColor Yellow
    Write-Host "Estimated size: ~50-100 MB" -ForegroundColor Yellow
    Write-Host ""
    
    if (-not $SkipConfirm) {
        $confirm = Read-Host "Continue? (Y/n)"
        if ($confirm -eq 'n' -or $confirm -eq 'N') { return }
    }
    
    Write-Host "Command: pip install ultralytics -i $TUNA_INDEX" -ForegroundColor Gray
    pip install ultralytics -i $TUNA_INDEX
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] ultralytics installed successfully" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] ultralytics installation failed" -ForegroundColor Red
        return $false
    }
    Write-Host ""
    return $true
}

# Execute installation
if ($Step -gt 0) {
    switch ($Step) {
        1 { Install-Step1 }
        2 { Install-Step2 }
        3 { Install-Step3 }
        4 { Install-Step4 }
        5 { Install-Step5 }
        default {
            Write-Host "[ERROR] Invalid step number. Please use 1-5" -ForegroundColor Red
            exit 1
        }
    }
} else {
    Write-Host "Running all 5 steps..." -ForegroundColor Yellow
    Write-Host ""
    
    if (-not $SkipConfirm) {
        $confirm = Read-Host "Start full installation? (Y/n)"
        if ($confirm -eq 'n' -or $confirm -eq 'N') {
            Write-Host "Installation cancelled" -ForegroundColor Yellow
            exit 0
        }
    }
    
    Install-Step1
    $result2 = Install-Step2
    if ($result2 -eq $false) { Write-Host "Installation stopped at step 2" -ForegroundColor Red; exit 1 }
    
    $result3 = Install-Step3
    if ($result3 -eq $false) { Write-Host "Installation stopped at step 3" -ForegroundColor Red; exit 1 }
    
    $result4 = Install-Step4
    if ($result4 -eq $false) { Write-Host "Installation stopped at step 4" -ForegroundColor Red; exit 1 }
    
    $result5 = Install-Step5
    if ($result5 -eq $false) { Write-Host "Installation stopped at step 5" -ForegroundColor Red; exit 1 }
}

# Completion
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run: .\start_cat_monitor.ps1" -ForegroundColor White
Write-Host "  2. Or: python -m cat_monitor.main" -ForegroundColor White
Write-Host ""
Write-Host "For help, see README.md" -ForegroundColor Cyan
Write-Host ""