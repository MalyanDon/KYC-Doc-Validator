"""
Explicit Feature Validator
Validates documents using explicit rules based on known PAN/Aadhaar characteristics
"""

import cv2
import numpy as np
import re
from typing import Dict, Tuple, Optional
from ocr_utils import extract_document_info


def validate_pan_format(text: str) -> Tuple[bool, float]:
    """
    Validate PAN number format: ABCDE1234F (5 letters, 4 digits, 1 letter)
    Returns: (is_valid, confidence)
    """
    # PAN format: 5 letters, 4 digits, 1 letter
    pan_pattern = r'[A-Z]{5}\d{4}[A-Z]{1}'
    matches = re.findall(pan_pattern, text.upper())
    
    if matches:
        return True, 1.0
    
    # Check for partial matches
    if re.search(r'[A-Z]{3,5}\d{3,4}[A-Z]?', text.upper()):
        return False, 0.5  # Partial match
    
    return False, 0.0


def validate_aadhaar_format(text: str) -> Tuple[bool, float]:
    """
    Validate Aadhaar number format: 12 digits (XXXX XXXX XXXX)
    Returns: (is_valid, confidence)
    """
    # Remove spaces and check for 12 digits
    digits_only = re.sub(r'\D', '', text)
    
    if len(digits_only) == 12:
        return True, 1.0
    
    if len(digits_only) >= 10:
        return False, 0.5  # Close but not exact
    
    return False, 0.0


def check_pan_keywords(text: str) -> Tuple[bool, float]:
    """
    Check for PAN card keywords
    Returns: (found_keywords, confidence)
    """
    pan_keywords = [
        'income tax',
        'tax department',
        'permanent account',
        'pan card',
        'government of india'
    ]
    
    text_lower = text.lower()
    found = sum(1 for keyword in pan_keywords if keyword in text_lower)
    
    if found >= 2:
        return True, 1.0
    elif found == 1:
        return True, 0.6
    else:
        return False, 0.0


def check_aadhaar_keywords(text: str) -> Tuple[bool, float]:
    """
    Check for Aadhaar card keywords
    Returns: (found_keywords, confidence)
    """
    aadhaar_keywords = [
        'government of india',
        'uidai',
        'aadhaar',
        'unique identification',
        'date of birth',
        'enrolment'
    ]
    
    text_lower = text.lower()
    found = sum(1 for keyword in aadhaar_keywords if keyword in text_lower)
    
    if found >= 2:
        return True, 1.0
    elif found == 1:
        return True, 0.6
    else:
        return False, 0.0


def validate_document_features(image: np.ndarray, ocr_text: str, predicted_type: str) -> Dict:
    """
    Validate document using explicit features
    
    Args:
        image: Input image
        ocr_text: Extracted OCR text
        predicted_type: Model's predicted type (PAN/Aadhaar/Other/Fake)
    
    Returns:
        Validation results with explicit feature checks
    """
    results = {
        'pan_validation': {},
        'aadhaar_validation': {},
        'recommended_type': predicted_type,
        'confidence_adjustment': 0.0,
        'explicit_features': {}
    }
    
    # PAN Card Validation
    pan_format_valid, pan_format_conf = validate_pan_format(ocr_text)
    pan_keywords_found, pan_keywords_conf = check_pan_keywords(ocr_text)
    
    results['pan_validation'] = {
        'pan_format_valid': pan_format_valid,
        'pan_format_confidence': pan_format_conf,
        'pan_keywords_found': pan_keywords_found,
        'pan_keywords_confidence': pan_keywords_conf,
        'overall_pan_score': (pan_format_conf + pan_keywords_conf) / 2
    }
    
    # Aadhaar Card Validation
    aadhaar_format_valid, aadhaar_format_conf = validate_aadhaar_format(ocr_text)
    aadhaar_keywords_found, aadhaar_keywords_conf = check_aadhaar_keywords(ocr_text)
    
    results['aadhaar_validation'] = {
        'aadhaar_format_valid': aadhaar_format_valid,
        'aadhaar_format_confidence': aadhaar_format_conf,
        'aadhaar_keywords_found': aadhaar_keywords_found,
        'aadhaar_keywords_confidence': aadhaar_keywords_conf,
        'overall_aadhaar_score': (aadhaar_format_conf + aadhaar_keywords_conf) / 2
    }
    
    # Determine recommended type based on explicit features
    pan_score = results['pan_validation']['overall_pan_score']
    aadhaar_score = results['aadhaar_validation']['overall_aadhaar_score']
    
    # If model predicted PAN but explicit features don't support it
    if predicted_type.lower() == 'pan':
        if pan_score < 0.3:
            # PAN features not found - likely not a PAN card
            results['recommended_type'] = 'Other'
            results['confidence_adjustment'] = -0.4
            results['explicit_features']['issue'] = 'PAN predicted but no PAN features found'
        elif pan_score >= 0.5:
            # PAN features found - supports prediction
            results['confidence_adjustment'] = +0.2
    
    # If model predicted Aadhaar but explicit features don't support it
    elif predicted_type.lower() == 'aadhaar':
        if aadhaar_score < 0.3:
            # Aadhaar features not found - likely not an Aadhaar card
            results['recommended_type'] = 'Other'
            results['confidence_adjustment'] = -0.4
            results['explicit_features']['issue'] = 'Aadhaar predicted but no Aadhaar features found'
        elif aadhaar_score >= 0.5:
            # Aadhaar features found - supports prediction
            results['confidence_adjustment'] = +0.2
    
    # If model predicted Other but explicit features suggest PAN/Aadhaar
    elif predicted_type.lower() == 'other':
        if pan_score > aadhaar_score and pan_score >= 0.5:
            results['recommended_type'] = 'PAN'
            results['confidence_adjustment'] = +0.3
            results['explicit_features']['suggestion'] = 'PAN features detected'
        elif aadhaar_score > pan_score and aadhaar_score >= 0.5:
            results['recommended_type'] = 'Aadhaar'
            results['confidence_adjustment'] = +0.3
            results['explicit_features']['suggestion'] = 'Aadhaar features detected'
    
    return results


def validate_with_explicit_features(image: np.ndarray, predicted_type: str, ocr_info: Dict) -> Dict:
    """
    Complete validation using explicit features
    
    Args:
        image: Input image
        predicted_type: Model's predicted type
        ocr_info: OCR extraction results
    
    Returns:
        Complete validation with explicit feature checks
    """
    ocr_text = ocr_info.get('raw_text', '')
    
    # Run explicit feature validation
    feature_validation = validate_document_features(image, ocr_text, predicted_type)
    
    # Extract PAN/Aadhaar numbers from OCR
    pan_number = ocr_info.get('pan_number', '')
    aadhaar_number = ocr_info.get('aadhaar_number', '')
    
    # Additional validation based on extracted numbers
    if pan_number:
        pan_format_valid, _ = validate_pan_format(pan_number)
        feature_validation['pan_validation']['extracted_pan_valid'] = pan_format_valid
    else:
        feature_validation['pan_validation']['extracted_pan_valid'] = False
    
    if aadhaar_number:
        aadhaar_format_valid, _ = validate_aadhaar_format(aadhaar_number)
        feature_validation['aadhaar_validation']['extracted_aadhaar_valid'] = aadhaar_format_valid
    else:
        feature_validation['aadhaar_validation']['extracted_aadhaar_valid'] = False
    
    return feature_validation

