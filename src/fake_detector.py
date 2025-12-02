"""
Fake Document Detection Module
Uses OpenCV for color analysis, edge detection, and tamper detection
Uses Pyzbar for QR code validation
"""

import cv2
import numpy as np
from pyzbar import pyzbar
from typing import Dict, List, Tuple
import re


def analyze_color_histogram(image: np.ndarray, doc_type: str = 'aadhaar') -> Dict:
    """
    Analyze color histogram to detect color mismatches
    Aadhaar cards typically have a blue tint
    """
    issues = []
    confidence = 1.0
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate histogram for hue channel
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    dominant_hue = None
    
    if doc_type.lower() == 'aadhaar':
        # CORRECTED: Aadhaar cards typically have WHITE background
        # Only Baal Aadhaar (children under 5) has blue background
        # Check if dominant hue is white (low saturation, high brightness)
        dominant_hue = int(np.argmax(hist_h))
        mean_saturation = np.mean(hsv[:, :, 1])
        mean_value = np.mean(hsv[:, :, 2])
        
        # White background: low saturation (<30) and high brightness (>200)
        # Blue background (Baal Aadhaar): hue 100-130, higher saturation
        is_white = mean_saturation < 30 and mean_value > 200
        is_blue = (100 <= dominant_hue <= 130) and mean_saturation > 50
        
        # Accept both white (standard) and blue (Baal Aadhaar)
        if not (is_white or is_blue):
            issues.append('color_mismatch')
            confidence -= 0.2  # Reduced penalty since both colors are valid
    
    # Check for plain white paper (high saturation in all channels)
    mean_saturation = np.mean(hsv[:, :, 1])
    mean_value = np.mean(hsv[:, :, 2])
    
    if mean_saturation < 20 and mean_value > 240:
        # Very white, low saturation - might be plain paper
        issues.append('plain_paper_detected')
        confidence -= 0.4
    
    return {
        'issues': issues,
        'confidence': max(0.0, confidence),
        'dominant_hue': dominant_hue,
        'mean_saturation': float(mean_saturation),
        'mean_value': float(mean_value)
    }


def detect_tampered_borders(image: np.ndarray) -> Dict:
    """
    Detect tampered borders using edge detection
    """
    issues = []
    confidence = 1.0
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Check border regions for unusual edge patterns
    h, w = edges.shape
    border_width = min(20, h // 10, w // 10)
    
    # Extract border regions
    top_border = edges[0:border_width, :]
    bottom_border = edges[h-border_width:h, :]
    left_border = edges[:, 0:border_width]
    right_border = edges[:, w-border_width:w]
    
    # Calculate edge density in borders
    border_edges = np.concatenate([
        top_border.flatten(),
        bottom_border.flatten(),
        left_border.flatten(),
        right_border.flatten()
    ])
    
    edge_density = np.sum(border_edges > 0) / len(border_edges)
    
    # Unusually high edge density might indicate tampering
    if edge_density > 0.3:
        issues.append('tampered_borders')
        confidence -= 0.2
    
    # Check for straight lines that might indicate cut/paste
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=min(h, w)//4, maxLineGap=10)
    
    if lines is not None and len(lines) > 10:
        # Too many long lines might indicate tampering
        issues.append('suspicious_lines')
        confidence -= 0.15
    
    return {
        'issues': issues,
        'confidence': max(0.0, confidence),
        'edge_density': float(edge_density),
        'line_count': len(lines) if lines is not None else 0
    }


def detect_handwritten_numbers(image: np.ndarray, ocr_text: str) -> Dict:
    """
    Detect handwritten numbers on white paper
    Uses OCR confidence and image analysis
    Enhanced with Magnum-Opus faulty image detection
    """
    issues = []
    confidence = 1.0
    variance = None
    mean_intensity = None
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Magnum-Opus faulty image detection: Check standard deviation
    std_dev = np.std(gray)
    if std_dev < 15.0:
        # Very uniform image - likely plain paper or blank
        issues.append('faulty_image_very_uniform')
        confidence -= 0.6
    
    # Check if image is all zeros (completely black/empty)
    if cv2.countNonZero(gray) == 0:
        issues.append('faulty_image_empty')
        confidence -= 0.8
    
    # Check if text contains numbers but image looks like plain paper
    has_numbers = bool(re.search(r'\d', ocr_text))
    
    if has_numbers:
        # Calculate variance - handwritten on plain paper has low variance
        variance = np.var(gray)
        
        # Plain paper with handwritten text typically has variance < 500
        if variance < 500:
            issues.append('handwritten_on_plain_paper')
            confidence -= 0.5
        
        # Check for uniform background
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        if mean_intensity > 200 and std_intensity < 30:
            issues.append('uniform_background_suspicious')
            confidence -= 0.3
    
    return {
        'issues': issues,
        'confidence': max(0.0, confidence),
        'variance': float(variance) if variance is not None else None,
        'mean_intensity': float(mean_intensity) if mean_intensity is not None else None,
        'std_dev': float(std_dev)
    }


def validate_qr_code(image: np.ndarray) -> Dict:
    """
    Validate QR code using Pyzbar
    """
    issues = []
    confidence = 1.0
    qr_data = None
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Decode QR codes
    decoded_objects = pyzbar.decode(gray)
    
    if not decoded_objects:
        # No QR code found - might be fake
        issues.append('no_qr_code')
        confidence -= 0.2
    else:
        # QR code found - extract data
        for obj in decoded_objects:
            if obj.type == 'QRCODE':
                qr_data = obj.data.decode('utf-8')
                break
        
        # Basic validation of QR data
        if qr_data:
            # Aadhaar QR typically contains encrypted data
            if len(qr_data) < 50:
                issues.append('invalid_qr_format')
                confidence -= 0.15
    
    return {
        'issues': issues,
        'confidence': max(0.0, confidence),
        'qr_found': len(decoded_objects) > 0,
        'qr_data': qr_data[:100] if qr_data else None  # Limit data length
    }


def detect_layout_tampering(image: np.ndarray) -> Dict:
    """
    Detect layout tampering by analyzing document structure
    """
    issues = []
    confidence = 1.0
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Use contour detection to find document structure
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        issues.append('no_structure_detected')
        confidence -= 0.3
        return {
            'issues': issues,
            'confidence': max(0.0, confidence),
            'contour_count': 0
        }
    
    # Find largest contour (should be document boundary)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    image_area = gray.shape[0] * gray.shape[1]
    
    # If largest contour is too small relative to image, might be tampered
    if area / image_area < 0.5:
        issues.append('incomplete_document')
        confidence -= 0.25
    
    # Check for rectangular shape (documents should be roughly rectangular)
    perimeter = cv2.arcLength(largest_contour, True)
    if perimeter > 0:
        extent = area / (perimeter * perimeter / 16)  # Rough rectangularity measure
        if extent < 0.3:
            issues.append('irregular_shape')
            confidence -= 0.2
    
    return {
        'issues': issues,
        'confidence': max(0.0, confidence),
        'contour_count': len(contours),
        'document_area_ratio': float(area / image_area) if image_area > 0 else 0.0
    }


def detect_pasted_photo_on_white_paper(image: np.ndarray, ocr_text: str) -> Dict:
    """
    Detect if someone pasted a photo on white paper with random numbers
    This is a common fake document scenario
    """
    issues = []
    confidence = 1.0
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # 1. Detect white paper background
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    is_white_paper = (mean_intensity > 200 and std_intensity < 40)
    
    if is_white_paper:
        issues.append('white_paper_background')
        confidence -= 0.3
        
        # 2. Detect photo-like regions (rectangular regions with different characteristics)
        # Use edge detection to find rectangular regions
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours that might be pasted photos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        photo_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Significant region
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if region has different characteristics than background
                region = gray[y:y+h, x:x+w]
                region_mean = np.mean(region)
                region_std = np.std(region)
                
                # Photo regions typically have different stats than white paper
                if abs(region_mean - mean_intensity) > 30 or region_std > std_intensity * 1.5:
                    photo_regions.append((x, y, w, h, region_mean, region_std))
        
        if len(photo_regions) > 0:
            issues.append('pasted_photo_detected')
            confidence -= 0.4
            
            # 3. Check for sharp edges around photo (indicates pasting)
            for x, y, w, h, _, _ in photo_regions:
                # Extract border around photo region
                border_region = np.concatenate([
                    gray[max(0, y-5):y, x:x+w].flatten(),  # Top border
                    gray[y+h:min(gray.shape[0], y+h+5), x:x+w].flatten(),  # Bottom border
                    gray[y:y+h, max(0, x-5):x].flatten(),  # Left border
                    gray[y:y+h, x+w:min(gray.shape[1], x+w+5)].flatten()  # Right border
                ])
                
                if len(border_region) > 0:
                    border_std = np.std(border_region)
                    # Sharp transitions indicate pasting
                    if border_std > 20:
                        issues.append('sharp_photo_edges_detected')
                        confidence -= 0.3
                        break
        
        # 4. Check if numbers look randomly placed (not in document structure)
        if ocr_text:
            # Real documents have structured text
            # Random numbers on white paper won't have structure
            lines = ocr_text.split('\n')
            non_empty_lines = [l.strip() for l in lines if l.strip()]
            
            # If we have numbers but very few structured lines, suspicious
            has_numbers = bool(re.search(r'\d', ocr_text))
            if has_numbers and len(non_empty_lines) < 3:
                issues.append('random_numbers_no_structure')
                confidence -= 0.3
    
    # 5. Check for missing document security features
    # Real documents have watermarks, patterns, etc.
    # White paper won't have these
    
    # Use frequency domain analysis to detect patterns
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    
    # Real documents have more complex frequency patterns
    freq_variance = np.var(magnitude_spectrum)
    if freq_variance < 2.0:  # Very simple frequency pattern
        issues.append('missing_security_patterns')
        confidence -= 0.2
    
    return {
        'issues': issues,
        'confidence': max(0.0, confidence),
        'is_white_paper': is_white_paper,
        'photo_regions_count': len(photo_regions) if is_white_paper else 0,
        'freq_variance': float(freq_variance)
    }


def detect_photo_tampering(image: np.ndarray) -> Dict:
    """
    Detect if the photo on the document has been tampered with
    (e.g., different photo than original, pasted photo)
    """
    issues = []
    confidence = 1.0
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # 1. Detect photo region (typically in top-left or center)
    h, w = gray.shape
    
    # Common photo locations in Indian ID documents
    photo_regions = [
        (int(w*0.05), int(h*0.15), int(w*0.25), int(h*0.35)),  # Top-left
        (int(w*0.35), int(h*0.15), int(w*0.55), int(h*0.35)),  # Top-center
    ]
    
    tampering_detected = False
    
    for x, y, x2, y2 in photo_regions:
        if x2 > w or y2 > h:
            continue
            
        photo_region = gray[y:y2, x:x2]
        if photo_region.size == 0:
            continue
        
        # Check for sharp edges around photo (indicates pasting)
        # Extract border
        border_width = 3
        border_pixels = []
        
        # Top border
        if y - border_width >= 0:
            border_pixels.extend(gray[y-border_width:y, x:x2].flatten())
        # Bottom border
        if y2 + border_width < h:
            border_pixels.extend(gray[y2:y2+border_width, x:x2].flatten())
        # Left border
        if x - border_width >= 0:
            border_pixels.extend(gray[y:y2, x-border_width:x].flatten())
        # Right border
        if x2 + border_width < w:
            border_pixels.extend(gray[y:y2, x2:x2+border_width].flatten())
        
        if len(border_pixels) > 0:
            border_std = np.std(border_pixels)
            photo_std = np.std(photo_region)
            
            # Sharp transition indicates tampering
            if border_std > photo_std * 1.5 and border_std > 25:
                tampering_detected = True
                break
    
    if tampering_detected:
        issues.append('photo_tampering_detected')
        confidence -= 0.5
    
    # 2. Check for inconsistent lighting (pasted photos often have different lighting)
    # This is a simplified check - real implementation would use more sophisticated methods
    photo_lighting = None
    for x, y, x2, y2 in photo_regions:
        if x2 > w or y2 > h:
            continue
        photo_region = gray[y:y2, x:x2]
        if photo_region.size > 0:
            photo_lighting = np.mean(photo_region)
            break
    
    if photo_lighting is not None:
        document_lighting = np.mean(gray)
        if abs(photo_lighting - document_lighting) > 40:
            issues.append('inconsistent_photo_lighting')
            confidence -= 0.3
    
    return {
        'issues': issues,
        'confidence': max(0.0, confidence),
        'tampering_detected': tampering_detected
    }


def comprehensive_fake_detection(image: np.ndarray, doc_type: str, ocr_text: str, use_layout_validation: bool = True) -> Dict:
    """
    Comprehensive fake detection combining all methods
    """
    all_issues = []
    confidence_scores = []
    
    # Color analysis
    color_result = analyze_color_histogram(image, doc_type)
    all_issues.extend(color_result['issues'])
    confidence_scores.append(color_result['confidence'])
    
    # Border tampering detection
    border_result = detect_tampered_borders(image)
    all_issues.extend(border_result['issues'])
    confidence_scores.append(border_result['confidence'])
    
    # Handwritten detection
    handwritten_result = detect_handwritten_numbers(image, ocr_text)
    all_issues.extend(handwritten_result['issues'])
    confidence_scores.append(handwritten_result['confidence'])
    
    # QR code validation
    qr_result = validate_qr_code(image)
    all_issues.extend(qr_result['issues'])
    confidence_scores.append(qr_result['confidence'])
    
    # Layout tampering
    layout_result = detect_layout_tampering(image)
    all_issues.extend(layout_result['issues'])
    confidence_scores.append(layout_result['confidence'])
    
    # NEW: Detect pasted photo on white paper (your specific scenario)
    pasted_photo_result = detect_pasted_photo_on_white_paper(image, ocr_text)
    all_issues.extend(pasted_photo_result['issues'])
    confidence_scores.append(pasted_photo_result['confidence'])
    
    # NEW: Detect photo tampering
    photo_tamper_result = detect_photo_tampering(image)
    all_issues.extend(photo_tamper_result['issues'])
    confidence_scores.append(photo_tamper_result['confidence'])
    
    # NEW: Layout-based validation (position checking)
    if use_layout_validation:
        try:
            from layout_validator import comprehensive_layout_validation
            layout_result = comprehensive_layout_validation(image, doc_type, ocr_text)
            all_issues.extend(layout_result['issues'])
            confidence_scores.append(layout_result['confidence'])
        except ImportError:
            # Layout validator not available, skip
            layout_result = None
    else:
        layout_result = None
    
    # NEW: Position-based detection using model predictions
    position_result = None
    try:
        from position_based_fake_detector import detect_fake_using_positions
        position_result = detect_fake_using_positions(image, doc_type)
        if position_result.get('is_fake', False):
            all_issues.extend(position_result.get('issues', []))
        confidence_scores.append(position_result.get('confidence', 1.0))
    except Exception as e:
        # Position detector not available or error, skip
        position_result = {'error': str(e)}
    
    # Calculate overall confidence (weighted average)
    overall_confidence = np.mean(confidence_scores)
    
    # Remove duplicate issues
    unique_issues = list(set(all_issues))
    
    return {
        'is_fake': overall_confidence < 0.7,
        'authenticity_score': float(overall_confidence),
        'issues': unique_issues,
        'detailed_results': {
            'color_analysis': color_result,
            'border_detection': border_result,
            'handwritten_detection': handwritten_result,
            'qr_validation': qr_result,
            'layout_analysis': layout_result,
            'pasted_photo_detection': pasted_photo_result,
            'photo_tampering': photo_tamper_result,
            'layout_validation': layout_result if layout_result else {},
            'position_based_detection': position_result if position_result else {}
        }
    }


if __name__ == "__main__":
    # Test fake detection
    print("Testing fake detection...")
    
    # Create a test image
    test_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
    cv2.rectangle(test_image, (10, 10), (390, 290), (0, 0, 0), 2)
    
    result = comprehensive_fake_detection(test_image, 'aadhaar', 'Test 1234 5678 9012')
    print(f"Detection result: {result}")

