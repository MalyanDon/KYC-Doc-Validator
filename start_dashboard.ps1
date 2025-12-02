# KYC Document Validator Dashboard Launcher
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "KYC Document Validator Dashboard" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Starting enhanced dashboard..." -ForegroundColor Green
Write-Host ""

# Activate virtual environment
& ".\venv\Scripts\Activate.ps1"

# Add Tesseract to PATH
$env:Path += ";C:\Program Files\Tesseract-OCR"

# Run Streamlit
Write-Host "Dashboard will open in your browser at http://localhost:8501" -ForegroundColor Yellow
Write-Host ""

streamlit run app/streamlit_app_enhanced.py

