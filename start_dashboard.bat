@echo off
echo ========================================
echo KYC Document Validator Dashboard
echo ========================================
echo.
echo Starting enhanced dashboard...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Set Tesseract path
set PATH=%PATH%;C:\Program Files\Tesseract-OCR

REM Run Streamlit
streamlit run app/streamlit_app_enhanced.py

pause

