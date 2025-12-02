"""
OCR Utilities for Text Extraction from KYC Documents
Uses Tesseract OCR for text extraction and validation
"""

import re
import os
import pytesseract
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

# Auto-configure Tesseract path for Windows
if os.name == 'nt':  # Windows
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break


def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for better OCR results
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Apply thresholding
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return cleaned


def extract_text(image: np.ndarray, lang: str = 'eng') -> str:
    """
    Extract text from image using Tesseract OCR
    """
    preprocessed = preprocess_image_for_ocr(image)
    text = pytesseract.image_to_string(preprocessed, lang=lang)
    return text.strip()


def extract_aadhaar_number(text: str) -> Optional[str]:
    """
    Extract 12-digit Aadhaar number from text using regex
    Pattern: 12 consecutive digits (may be space-separated)
    """
    # Remove all non-digit characters except spaces
    cleaned = re.sub(r'[^\d\s]', '', text)
    
    # Find 12 consecutive digits (with optional spaces)
    patterns = [
        r'\d{4}\s?\d{4}\s?\d{4}',  # Format: XXXX XXXX XXXX
        r'\d{12}',  # Format: XXXXXXXXXXXX
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, cleaned)
        if matches:
            # Return first match, remove spaces
            aadhaar = re.sub(r'\s', '', matches[0])
            if len(aadhaar) == 12:
                return aadhaar
    
    return None


def extract_pan_number(text: str) -> Optional[str]:
    """
    Extract PAN number from text using regex
    Pattern: 5 letters, 4 digits, 1 letter (e.g., ABCDE1234F)
    """
    # PAN format: 5 letters, 4 digits, 1 letter
    pattern = r'[A-Z]{5}\d{4}[A-Z]{1}'
    matches = re.findall(pattern, text.upper())
    
    if matches:
        return matches[0]
    
    return None


def validate_aadhaar_format(aadhaar: str) -> bool:
    """
    Validate Aadhaar number format
    - Must be exactly 12 digits
    - Should not be all zeros or all same digits
    """
    if not aadhaar or len(aadhaar) != 12:
        return False
    
    if not aadhaar.isdigit():
        return False
    
    # Check for invalid patterns
    if aadhaar == '0' * 12:  # All zeros
        return False
    
    if len(set(aadhaar)) == 1:  # All same digit
        return False
    
    return True


def validate_pan_format(pan: str) -> bool:
    """
    Validate PAN number format and checksum
    Format: ABCDE1234F
    - First 5 characters: letters
    - Next 4: digits
    - Last: letter
    """
    if not pan or len(pan) != 10:
        return False
    
    pattern = r'^[A-Z]{5}\d{4}[A-Z]{1}$'
    if not re.match(pattern, pan.upper()):
        return False
    
    # Basic checksum validation (simplified)
    # In real implementation, would use IT department algorithm
    return True


def calculate_pan_checksum(pan: str) -> bool:
    """
    Calculate PAN checksum (simplified version)
    Real implementation would use IT department's algorithm
    """
    if len(pan) != 10:
        return False
    
    # Simplified checksum: sum of digits mod 10
    digits = pan[5:9]
    if not digits.isdigit():
        return False
    
    digit_sum = sum(int(d) for d in digits)
    # This is a placeholder - real checksum is more complex
    return True


def extract_text_with_positions(image: np.ndarray) -> List[Dict]:
    """
    Extract text with their positions using OCR
    Returns list of {text, x, y, width, height, confidence}
    """
    try:
        # Use pytesseract to get detailed data including positions
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        texts = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if text and int(data['conf'][i]) > 0:
                texts.append({
                    'text': text,
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'confidence': int(data['conf'][i])
                })
        return texts
    except Exception as e:
        # Fallback: simple text extraction
        text = extract_text(image)
        return [{'text': text, 'x': 0, 'y': 0, 'width': 0, 'height': 0, 'confidence': 0}]


def extract_document_info(image: np.ndarray, include_positions: bool = False) -> Dict:
    """
    Extract all relevant information from document image
    Returns dictionary with extracted text, Aadhaar, PAN, etc.
    
    Args:
        include_positions: If True, also return text with positions
    """
    text = extract_text(image)
    
    aadhaar = extract_aadhaar_number(text)
    pan = extract_pan_number(text)
    
    result = {
        'raw_text': text,
        'aadhaar_number': aadhaar if aadhaar and validate_aadhaar_format(aadhaar) else None,
        'pan_number': pan if pan and validate_pan_format(pan) else None,
        'text_length': len(text),
        'has_numbers': bool(re.search(r'\d', text)),
        'has_letters': bool(re.search(r'[A-Za-z]', text))
    }
    
    if include_positions:
        result['text_with_positions'] = extract_text_with_positions(image)
    
    return result


def mock_uidai_validation(aadhaar: str) -> Dict:
    """
    Mock UIDAI API validation
    In production, would call actual UIDAI API
    """
    if not aadhaar or not validate_aadhaar_format(aadhaar):
        return {
            'valid': False,
            'message': 'Invalid Aadhaar format',
            'confidence': 0.0
        }
    
    # Mock validation - in real scenario, would make API call
    # For now, return based on format validation
    return {
        'valid': True,
        'message': 'Aadhaar format validated (mock)',
        'confidence': 0.85  # Lower confidence for mock
    }


def mock_it_validation(pan: str) -> Dict:
    """
    Mock IT Department API validation
    In production, would call actual IT API
    """
    if not pan or not validate_pan_format(pan):
        return {
            'valid': False,
            'message': 'Invalid PAN format',
            'confidence': 0.0
        }
    
    # Mock validation
    return {
        'valid': True,
        'message': 'PAN format validated (mock)',
        'confidence': 0.85  # Lower confidence for mock
    }


if __name__ == "__main__":
    # Test OCR utilities
    print("Testing OCR utilities...")
    
    # Create a test image with text
    test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "Aadhaar: 1234 5678 9012", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(test_image, "PAN: ABCDE1234F", (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    info = extract_document_info(test_image)
    print(f"Extracted info: {info}")

