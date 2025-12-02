"""
Layout Validator - Position-based Document Validation
Checks if photo, text, and elements are in correct positions for real documents
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import pytesseract
from dataclasses import dataclass


@dataclass
class DocumentLayout:
    """Expected layout structure for different document types"""
    doc_type: str
    photo_region: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max) as ratios
    text_regions: List[Tuple[str, float, float, float, float]]  # (label, x_min, y_min, x_max, y_max)
    security_features: List[str]  # Expected security features


# Default layouts (will be replaced by learned positions if available)
def get_default_layout(doc_type: str) -> DocumentLayout:
    """Get default layout - can be overridden by learned positions"""
    if doc_type.lower() == 'aadhaar':
        return DocumentLayout(
            doc_type='aadhaar',
            photo_region=(0.05, 0.15, 0.30, 0.40),  # Photo typically in top-left
            text_regions=[
                ('name', 0.35, 0.20, 0.95, 0.30),
                ('dob', 0.35, 0.30, 0.70, 0.40),
                ('gender', 0.35, 0.40, 0.60, 0.50),
                ('aadhaar_number', 0.35, 0.50, 0.95, 0.60),
                ('address', 0.05, 0.60, 0.95, 0.85),
            ],
            security_features=['watermark', 'qr_code', 'logo']
        )
    else:  # PAN
        return DocumentLayout(
            doc_type='pan',
            photo_region=(0.70, 0.15, 0.95, 0.40),  # Photo typically in top-right
            text_regions=[
                ('name', 0.05, 0.20, 0.65, 0.30),
                ('father_name', 0.05, 0.30, 0.65, 0.40),
                ('dob', 0.05, 0.40, 0.40, 0.50),
                ('pan_number', 0.05, 0.50, 0.60, 0.60),
                ('signature', 0.05, 0.60, 0.40, 0.85),
            ],
            security_features=['hologram', 'logo', 'security_pattern']
        )


# Try to load learned positions, fallback to defaults
def load_layout(doc_type: str) -> DocumentLayout:
    """Load layout - tries learned positions first, then defaults"""
    try:
        from trainable_layout_detector import LayoutPositionLearner
        learner = LayoutPositionLearner(doc_type=doc_type)
        
        # Try to load learned positions
        learned_file = f'models/learned_{doc_type}_positions.json'
        if os.path.exists(learned_file):
            learner.load_learned_positions(learned_file)
            learned = learner.get_positions()
            
            # Convert to DocumentLayout
            text_regions = []
            for label, pos in learned.get('text_regions', {}).items():
                text_regions.append((label, pos[0], pos[1], pos[2], pos[3]))
            
            return DocumentLayout(
                doc_type=doc_type,
                photo_region=learned.get('photo_region', (0.05, 0.15, 0.30, 0.40)),
                text_regions=text_regions,
                security_features=['watermark', 'qr_code', 'logo']  # Default
            )
    except:
        pass
    
    # Fallback to default
    return get_default_layout(doc_type)


# For backward compatibility
AADHAAR_LAYOUT = get_default_layout('aadhaar')
PAN_LAYOUT = get_default_layout('pan')


def detect_photo_region(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect photo region in document using face detection and region analysis
    Returns: (x, y, width, height) or None
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape
    
    # Method 1: Use face detection (if available)
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get largest face (likely the main photo)
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, fw, fh = largest_face
            
            # Expand region to include full photo (photos are usually larger than just face)
            photo_margin = 20
            x = max(0, x - photo_margin)
            y = max(0, y - photo_margin)
            width = min(w - x, fw + 2 * photo_margin)
            height = min(h - y, fh + 2 * photo_margin)
            
            return (x, y, width, height)
    except:
        pass
    
    # Method 2: Detect rectangular regions with different characteristics
    # Find regions that look like photos (rectangular, different from background)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    photo_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 5000 < area < (h * w * 0.3):  # Reasonable photo size
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            
            # Check aspect ratio (photos are usually roughly square or portrait)
            aspect_ratio = h_rect / w_rect if w_rect > 0 else 0
            if 0.8 < aspect_ratio < 1.5:
                # Check if region has different characteristics
                region = gray[y:y+h_rect, x:x+w_rect]
                if region.size > 0:
                    region_std = np.std(region)
                    if region_std > 20:  # Has variation (not uniform)
                        photo_candidates.append((x, y, w_rect, h_rect, area))
    
    if photo_candidates:
        # Return largest candidate
        largest = max(photo_candidates, key=lambda c: c[4])
        return (largest[0], largest[1], largest[2], largest[3])
    
    return None


def validate_photo_position(image: np.ndarray, doc_type: str, detected_photo: Optional[Tuple[int, int, int, int]]) -> Dict:
    """
    Validate if photo is in the correct position for the document type
    """
    issues = []
    confidence = 1.0
    
    if detected_photo is None:
        issues.append('no_photo_detected')
        confidence -= 0.5
        return {
            'issues': issues,
            'confidence': max(0.0, confidence),
            'position_valid': False,
            'expected_position': None,
            'actual_position': None
        }
    
    h, w = image.shape[:2]
    x, y, photo_w, photo_h = detected_photo
    
    # Get expected layout (try learned, fallback to default)
    layout = load_layout(doc_type.lower())
    
    if layout is None:
        return {
            'issues': ['unknown_document_type'],
            'confidence': 0.5,
            'position_valid': False,
            'expected_position': None,
            'actual_position': (x/w, y/h, (x+photo_w)/w, (y+photo_h)/h)
        }
    
    # Convert detected position to normalized coordinates
    detected_normalized = (x/w, y/h, (x+photo_w)/w, (y+photo_h)/h)
    expected_normalized = layout.photo_region
    
    # Calculate overlap
    overlap_x_min = max(detected_normalized[0], expected_normalized[0])
    overlap_y_min = max(detected_normalized[1], expected_normalized[1])
    overlap_x_max = min(detected_normalized[2], expected_normalized[2])
    overlap_y_max = min(detected_normalized[3], expected_normalized[3])
    
    if overlap_x_max > overlap_x_min and overlap_y_max > overlap_y_min:
        overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
        detected_area = (detected_normalized[2] - detected_normalized[0]) * (detected_normalized[3] - detected_normalized[1])
        overlap_ratio = overlap_area / detected_area if detected_area > 0 else 0
        
        if overlap_ratio < 0.5:  # Less than 50% overlap
            issues.append('photo_wrong_position')
            confidence -= 0.4
            position_valid = False
        else:
            position_valid = True
    else:
        issues.append('photo_wrong_position')
        confidence -= 0.5
        position_valid = False
    
    return {
        'issues': issues,
        'confidence': max(0.0, confidence),
        'position_valid': position_valid,
        'expected_position': expected_normalized,
        'actual_position': detected_normalized,
        'overlap_ratio': overlap_ratio if 'overlap_ratio' in locals() else 0.0
    }


def extract_text_with_positions(image: np.ndarray) -> List[Dict]:
    """
    Extract text with their positions using OCR
    Returns list of {text, x, y, width, height, confidence}
    """
    # Use pytesseract to get detailed data including positions
    try:
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
    except:
        # Fallback: simple text extraction
        text = pytesseract.image_to_string(image)
        return [{'text': text, 'x': 0, 'y': 0, 'width': 0, 'height': 0, 'confidence': 0}]


def validate_text_positions(image: np.ndarray, doc_type: str, extracted_texts: List[Dict]) -> Dict:
    """
    Validate if text elements are in correct positions
    """
    issues = []
    confidence = 1.0
    
    h, w = image.shape[:2]
    
    # Get expected layout (try learned, fallback to default)
    layout = load_layout(doc_type.lower())
    
    if layout is None:
        return {
            'issues': ['unknown_document_type'],
            'confidence': 0.5,
            'text_positions_valid': False
        }
    
    # Check each expected text region
    position_errors = []
    for label, x_min, y_min, x_max, y_max in layout.text_regions:
        # Convert to pixel coordinates
        px_min = int(x_min * w)
        py_min = int(y_min * h)
        px_max = int(x_max * w)
        py_max = int(y_max * h)
        
        # Find text in this region
        found_text = False
        for text_item in extracted_texts:
            tx = text_item['x']
            ty = text_item['y']
            tw = text_item['width']
            th = text_item['height']
            
            # Check if text overlaps with expected region
            if (px_min <= tx + tw/2 <= px_max and py_min <= ty + th/2 <= py_max):
                found_text = True
                break
        
        if not found_text:
            position_errors.append(f'missing_text_{label}')
            confidence -= 0.1
    
    if position_errors:
        issues.extend(position_errors)
        issues.append('text_position_mismatch')
    
    return {
        'issues': issues,
        'confidence': max(0.0, confidence),
        'text_positions_valid': len(issues) == 0,
        'position_errors': position_errors
    }


def verify_photo_authenticity(image: np.ndarray, photo_region: Tuple[int, int, int, int]) -> Dict:
    """
    Verify if the photo is authentic (not pasted, not tampered)
    Uses multiple techniques to detect photo tampering
    """
    issues = []
    confidence = 1.0
    
    x, y, w, h = photo_region
    h_img, w_img = image.shape[:2]
    
    # Extract photo region
    photo = image[y:y+h, x:x+w]
    if photo.size == 0:
        return {
            'issues': ['invalid_photo_region'],
            'confidence': 0.0,
            'is_authentic': False
        }
    
    gray_photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY) if len(photo.shape) == 3 else photo
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # 1. Check for sharp edges around photo (pasting indicator)
    border_width = 5
    border_pixels = []
    
    # Top border
    if y - border_width >= 0:
        border_pixels.extend(gray_full[y-border_width:y, max(0, x-border_width):min(w_img, x+w+border_width)].flatten())
    # Bottom border
    if y + h + border_width < h_img:
        border_pixels.extend(gray_full[y+h:y+h+border_width, max(0, x-border_width):min(w_img, x+w+border_width)].flatten())
    # Left border
    if x - border_width >= 0:
        border_pixels.extend(gray_full[max(0, y-border_width):min(h_img, y+h+border_width), x-border_width:x].flatten())
    # Right border
    if x + w + border_width < w_img:
        border_pixels.extend(gray_full[max(0, y-border_width):min(h_img, y+h+border_width), x+w:x+w+border_width].flatten())
    
    if len(border_pixels) > 0:
        border_std = np.std(border_pixels)
        photo_std = np.std(gray_photo)
        
        # Sharp transition indicates pasting
        if border_std > photo_std * 1.5 and border_std > 25:
            issues.append('sharp_edges_around_photo')
            confidence -= 0.4
    
    # 2. Check lighting consistency
    photo_mean = np.mean(gray_photo)
    document_mean = np.mean(gray_full)
    
    if abs(photo_mean - document_mean) > 40:
        issues.append('inconsistent_lighting')
        confidence -= 0.3
    
    # 3. Check for compression artifacts (pasted photos often have different compression)
    # Use frequency domain analysis
    f_transform = np.fft.fft2(gray_photo)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    
    photo_freq_var = np.var(magnitude_spectrum)
    
    # Get frequency pattern of document background
    bg_region = gray_full[0:min(50, h_img), 0:min(50, w_img)]
    if bg_region.size > 0:
        f_bg = np.fft.fft2(bg_region)
        f_bg_shift = np.fft.fftshift(f_bg)
        bg_magnitude = np.log(np.abs(f_bg_shift) + 1)
        bg_freq_var = np.var(bg_magnitude)
        
        # Very different frequency patterns indicate different sources
        if abs(photo_freq_var - bg_freq_var) > 2.0:
            issues.append('compression_artifact_mismatch')
            confidence -= 0.2
    
    # 4. Check photo quality (real photos should have reasonable quality)
    photo_quality_score = np.std(gray_photo) / np.mean(gray_photo) if np.mean(gray_photo) > 0 else 0
    
    if photo_quality_score < 0.1:  # Too uniform
        issues.append('low_photo_quality')
        confidence -= 0.2
    
    is_authentic = len(issues) == 0
    
    return {
        'issues': issues,
        'confidence': max(0.0, confidence),
        'is_authentic': is_authentic,
        'photo_quality_score': float(photo_quality_score),
        'lighting_diff': float(abs(photo_mean - document_mean))
    }


def comprehensive_layout_validation(image: np.ndarray, doc_type: str, ocr_text: str) -> Dict:
    """
    Comprehensive layout validation combining all checks
    """
    all_issues = []
    confidence_scores = []
    
    # 1. Detect photo region
    photo_region = detect_photo_region(image)
    
    # 2. Validate photo position
    position_result = validate_photo_position(image, doc_type, photo_region)
    all_issues.extend(position_result['issues'])
    confidence_scores.append(position_result['confidence'])
    
    # 3. Verify photo authenticity
    if photo_region:
        photo_auth_result = verify_photo_authenticity(image, photo_region)
        all_issues.extend(photo_auth_result['issues'])
        confidence_scores.append(photo_auth_result['confidence'])
    else:
        photo_auth_result = {'is_authentic': False, 'confidence': 0.5}
        confidence_scores.append(0.5)
    
    # 4. Extract text with positions
    extracted_texts = extract_text_with_positions(image)
    
    # 5. Validate text positions
    text_position_result = validate_text_positions(image, doc_type, extracted_texts)
    all_issues.extend(text_position_result['issues'])
    confidence_scores.append(text_position_result['confidence'])
    
    # Calculate overall confidence
    overall_confidence = np.mean(confidence_scores)
    
    # Remove duplicates
    unique_issues = list(set(all_issues))
    
    return {
        'is_valid': overall_confidence > 0.7,
        'confidence': float(overall_confidence),
        'issues': unique_issues,
        'photo_detected': photo_region is not None,
        'photo_position_valid': position_result.get('position_valid', False),
        'photo_authentic': photo_auth_result.get('is_authentic', False),
        'text_positions_valid': text_position_result.get('text_positions_valid', False),
        'detailed_results': {
            'photo_position': position_result,
            'photo_authenticity': photo_auth_result,
            'text_positions': text_position_result,
            'extracted_texts_count': len(extracted_texts)
        }
    }


if __name__ == "__main__":
    # Test
    import numpy as np
    test_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    result = comprehensive_layout_validation(test_image, 'aadhaar', 'Test text')
    print(f"Layout validation result: {result}")

