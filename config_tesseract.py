"""
Tesseract OCR Configuration
Sets the Tesseract executable path for pytesseract
Run this once after installing Tesseract OCR
"""

import os
import sys

# Tesseract installation path (Windows default)
TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
]

def configure_tesseract():
    """Configure pytesseract to use the correct Tesseract path"""
    import pytesseract
    
    # Check if already configured
    if hasattr(pytesseract.pytesseract, 'tesseract_cmd') and pytesseract.pytesseract.tesseract_cmd:
        if os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            print(f"[OK] Tesseract already configured: {pytesseract.pytesseract.tesseract_cmd}")
            return True
    
    # Try to find Tesseract
    for path in TESSERACT_PATHS:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"[OK] Tesseract configured: {path}")
            return True
    
    # Try to find in PATH
    import shutil
    tesseract_path = shutil.which('tesseract')
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"[OK] Tesseract found in PATH: {tesseract_path}")
        return True
    
    print("[ERROR] Tesseract not found. Please install Tesseract OCR.")
    print("   Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    return False

if __name__ == "__main__":
    configure_tesseract()

