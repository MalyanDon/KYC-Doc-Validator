"""
YOLO-Based Document Field Extraction
Uses YOLOv8 to detect and extract document fields
Integrates with our KYC validator
"""

import os
import cv2
import numpy as np
import pytesseract
import re
from typing import Dict, List, Tuple, Optional
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics not installed. Install with: pip install ultralytics")


class YOLODocumentExtractor:
    """
    Extract document information using YOLO object detection
    More accurate than full-document OCR
    """
    
    def __init__(self, model_path: str = 'models/yolov8_trained_model.pt', confidence: float = 0.5):
        """
        Initialize YOLO extractor
        
        Args:
            model_path: Path to trained YOLO model
            confidence: Confidence threshold for detections
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
        
        self.model = None
        self.confidence = confidence
        
        if model_path and os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                print(f"✅ Loaded YOLO model from {model_path}")
            except Exception as e:
                print(f"⚠️  Could not load YOLO model: {e}")
                print("💡 Will use fallback OCR extraction")
        else:
            print(f"⚠️  YOLO model not found at {model_path}")
            print("💡 Will use fallback OCR extraction")
    
    def extract_fields(self, image: np.ndarray) -> Dict:
        """
        Extract document fields using YOLO detection
        
        Returns:
            Dictionary with extracted fields and their positions
        """
        if self.model is None:
            return self._fallback_extraction(image)
        
        # Resize for YOLO (typically 640x640)
        original_shape = image.shape[:2]
        resized = cv2.resize(image, (640, 640))
        
        # Predict
        results = self.model.predict(source=resized, conf=self.confidence, verbose=0)
        
        extracted_info = {
            "GENDER": None,
            "AADHAR_NUMBER": None,
            "NAME": None,
            "DATE_OF_BIRTH": None,
            "fields_with_positions": []
        }
        
        # Process each detection
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                label_id = int(labels[i])
                confidence = float(confidences[i])
                label_name = result.names[label_id]
                
                # Scale coordinates back to original image size
                scale_x = original_shape[1] / 640
                scale_y = original_shape[0] / 640
                
                orig_x1 = int(x1 * scale_x)
                orig_y1 = int(y1 * scale_y)
                orig_x2 = int(x2 * scale_x)
                orig_y2 = int(y2 * scale_y)
                
                # Crop region
                cropped = resized[y1:y2, x1:x2]
                
                # Preprocess for OCR
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                
                # OCR with appropriate config
                ocr_config = "--psm 7 --oem 1" if "DATE" in label_name.upper() else "--psm 6"
                extracted_text = pytesseract.image_to_string(thresh, config=ocr_config).strip()
                
                # Clean text based on field type
                cleaned_text = self._clean_text(extracted_text, label_name)
                
                # Store extracted info
                field_key = self._get_field_key(label_name)
                if field_key:
                    extracted_info[field_key] = cleaned_text
                
                # Store position info
                extracted_info["fields_with_positions"].append({
                    'field': label_name,
                    'text': cleaned_text,
                    'position': (orig_x1, orig_y1, orig_x2, orig_y2),
                    'confidence': confidence
                })
        
        return extracted_info
    
    def _clean_text(self, text: str, field_type: str) -> str:
        """Clean extracted text based on field type"""
        if not text:
            return None
        
        field_upper = field_type.upper()
        
        if "AADHAR" in field_upper or "NUMBER" in field_upper:
            # Extract only digits
            cleaned = re.sub(r'\D', '', text)
            return cleaned if len(cleaned) == 12 else None
        
        elif "DATE" in field_upper or "DOB" in field_upper:
            # Extract date pattern
            date_match = re.search(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b', text)
            if date_match:
                return date_match.group(0)
            # Try other date formats
            date_match = re.search(r'\b\d{4}[/-]\d{2}[/-]\d{2}\b', text)
            return date_match.group(0) if date_match else text.strip()
        
        elif "GENDER" in field_upper:
            # Extract gender
            text_lower = text.lower()
            if 'male' in text_lower or 'm' in text_lower:
                return 'Male'
            elif 'female' in text_lower or 'f' in text_lower:
                return 'Female'
            return text.strip()
        
        else:
            # Name or other text - just clean whitespace
            return ' '.join(text.split())
    
    def _get_field_key(self, label: str) -> Optional[str]:
        """Map YOLO label to our field key"""
        label_upper = label.upper()
        
        if "NAME" in label_upper:
            return "NAME"
        elif "AADHAR" in label_upper or "AADHAAR" in label_upper:
            return "AADHAR_NUMBER"
        elif "DATE" in label_upper or "DOB" in label_upper:
            return "DATE_OF_BIRTH"
        elif "GENDER" in label_upper:
            return "GENDER"
        
        return None
    
    def _fallback_extraction(self, image: np.ndarray) -> Dict:
        """Fallback to basic OCR if YOLO not available"""
        from ocr_utils import extract_document_info
        
        info = extract_document_info(image, include_positions=True)
        
        return {
            "NAME": None,  # Would need to parse from raw_text
            "AADHAR_NUMBER": info.get('aadhaar_number'),
            "DATE_OF_BIRTH": None,  # Would need to parse from raw_text
            "GENDER": None,  # Would need to parse from raw_text
            "fields_with_positions": info.get('text_with_positions', [])
        }
    
    def visualize_detections(self, image: np.ndarray, save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize YOLO detections on image
        """
        if self.model is None:
            return image
        
        extracted = self.extract_fields(image)
        
        vis_image = image.copy()
        
        for field_info in extracted.get("fields_with_positions", []):
            x1, y1, x2, y2 = field_info['position']
            label = field_info['field']
            text = field_info['text']
            conf = field_info['confidence']
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label_text = f"{label}: {text} ({conf:.2f})"
            cv2.putText(vis_image, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image


def integrate_yolo_with_layout_validation(image: np.ndarray, doc_type: str, 
                                        yolo_model_path: Optional[str] = None) -> Dict:
    """
    Integrate YOLO extraction with layout validation
    Combines best of both approaches
    """
    from layout_validator import comprehensive_layout_validation
    
    # Extract using YOLO
    yolo_extractor = None
    yolo_extracted = {}
    
    if yolo_model_path and os.path.exists(yolo_model_path):
        try:
            yolo_extractor = YOLODocumentExtractor(yolo_model_path)
            yolo_extracted = yolo_extractor.extract_fields(image)
        except Exception as e:
            print(f"⚠️  YOLO extraction failed: {e}")
    
    # Get OCR text for layout validation
    from ocr_utils import extract_text
    ocr_text = extract_text(image)
    
    # Layout validation
    layout_result = comprehensive_layout_validation(image, doc_type, ocr_text)
    
    # Combine results
    combined_result = {
        'yolo_extraction': yolo_extracted,
        'layout_validation': layout_result,
        'extracted_fields': {
            'name': yolo_extracted.get('NAME'),
            'aadhaar_number': yolo_extracted.get('AADHAR_NUMBER'),
            'dob': yolo_extracted.get('DATE_OF_BIRTH'),
            'gender': yolo_extracted.get('GENDER')
        },
        'field_positions': yolo_extracted.get('fields_with_positions', []),
        'layout_valid': layout_result.get('is_valid', False),
        'authenticity_score': layout_result.get('confidence', 0.0)
    }
    
    return combined_result


if __name__ == "__main__":
    import os
    
    # Example usage
    if YOLO_AVAILABLE:
        print("Testing YOLO extractor...")
        
        # Check if model exists
        model_path = 'models/yolov8_trained_model.pt'
        if os.path.exists(model_path):
            extractor = YOLODocumentExtractor(model_path)
            
            # Test image
            test_image = np.ones((800, 600, 3), dtype=np.uint8) * 255
            result = extractor.extract_fields(test_image)
            print(f"Extracted: {result}")
        else:
            print(f"⚠️  YOLO model not found at {model_path}")
            print("💡 Train a YOLO model first or use fallback extraction")
    else:
        print("⚠️  ultralytics not installed")
        print("💡 Install with: pip install ultralytics")

