"""
Position-Based Fake Detection using Model Predictions
Uses the enhanced model's position predictions to detect fake documents
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
import json
from typing import Dict, Tuple, Optional
from pathlib import Path

from models_enhanced import create_enhanced_ensemble_model, compile_enhanced_model
import tensorflow as tf
from document_boundary_detector import (
    detect_document_boundaries,
    crop_to_boundaries,
    normalize_position_to_boundaries,
    draw_boundaries
)


def load_learned_positions(doc_type: str) -> Optional[Dict]:
    """Load learned positions from JSON file"""
    positions_file = f'models/learned_{doc_type.lower()}_positions.json'
    if not os.path.exists(positions_file):
        return None
    
    with open(positions_file, 'r') as f:
        data = json.load(f)
    return data


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (150, 150)) -> np.ndarray:
    """Preprocess image for model input"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    return image


def predict_positions(model, image: np.ndarray) -> np.ndarray:
    """Use model to predict element positions"""
    preprocessed = preprocess_image(image)
    input_batch = np.expand_dims(preprocessed, axis=0)
    
    predictions = model.predict(input_batch, verbose=0)
    # Third output is positions
    positions = predictions[2][0]  # Shape: (16,) - 4 elements * 4 coordinates
    
    return positions


def parse_positions(positions_array: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Parse position array into structured format
    Positions array: [photo_x_min, photo_y_min, photo_x_max, photo_y_max,
                     name_x_min, name_y_min, name_x_max, name_y_max,
                     dob_x_min, dob_y_min, dob_x_max, dob_y_max,
                     doc_num_x_min, doc_num_y_min, doc_num_x_max, doc_num_y_max]
    """
    return {
        'photo': positions_array[0:4],
        'name': positions_array[4:8],
        'dob': positions_array[8:12],
        'document_number': positions_array[12:16]
    }


def calculate_position_deviation(predicted: np.ndarray, learned_mean: np.ndarray, 
                                 learned_std: np.ndarray, threshold_sigma: float = 2.0) -> Tuple[float, bool]:
    """
    Calculate how much predicted positions deviate from learned positions
    Returns: (deviation_score, is_suspicious)
    """
    # Calculate z-scores for each coordinate
    z_scores = np.abs((predicted - learned_mean) / (learned_std + 1e-6))
    
    # Average z-score across all coordinates
    avg_z_score = np.mean(z_scores)
    
    # Check if any coordinate is beyond threshold
    max_z_score = np.max(z_scores)
    is_suspicious = max_z_score > threshold_sigma
    
    return float(avg_z_score), is_suspicious


def calculate_overlap_ratio(predicted: np.ndarray, learned_mean: np.ndarray) -> float:
    """
    Calculate overlap ratio between predicted and learned positions
    Returns: overlap ratio (0-1)
    """
    # Extract coordinates
    pred_x_min, pred_y_min, pred_x_max, pred_y_max = predicted
    learned_x_min, learned_y_min, learned_x_max, learned_y_max = learned_mean
    
    # Calculate overlap region
    overlap_x_min = max(pred_x_min, learned_x_min)
    overlap_y_min = max(pred_y_min, learned_y_min)
    overlap_x_max = min(pred_x_max, learned_x_max)
    overlap_y_max = min(pred_y_max, learned_y_max)
    
    # Check if there's overlap
    if overlap_x_max <= overlap_x_min or overlap_y_max <= overlap_y_min:
        return 0.0
    
    # Calculate areas
    overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
    pred_area = (pred_x_max - pred_x_min) * (pred_y_max - pred_y_min)
    
    if pred_area == 0:
        return 0.0
    
    overlap_ratio = overlap_area / pred_area
    return float(overlap_ratio)


def detect_fake_using_positions(image: np.ndarray, doc_type: str, 
                                model: Optional[tf.keras.Model] = None,
                                threshold_sigma: float = 2.0,
                                min_overlap: float = 0.5,
                                use_boundary_detection: bool = True) -> Dict:
    """
    Detect fake documents using position predictions from the model
    
    Args:
        image: Input image (BGR format)
        doc_type: Document type ('aadhaar' or 'pan')
        model: Trained enhanced model (if None, will load from file)
        threshold_sigma: Number of standard deviations allowed before flagging as suspicious
        min_overlap: Minimum overlap ratio required (0-1)
        use_boundary_detection: If True, detect document boundaries first, then normalize positions
    
    Returns:
        Dictionary with detection results
    """
    issues = []
    confidence = 1.0
    position_details = {}
    boundary_info = None
    
    # Step 1: Detect document boundaries (NEW!)
    if use_boundary_detection:
        try:
            boundary_info = detect_document_boundaries(image, doc_type)
            if boundary_info['confidence'] > 0.5:
                # Crop to document region
                cropped_image = crop_to_boundaries(image, boundary_info['boundaries'])
                original_size = (image.shape[1], image.shape[0])  # (width, height)
            else:
                # Low confidence - use full image
                cropped_image = image
                original_size = (image.shape[1], image.shape[0])
                boundary_info = None
        except Exception as e:
            # Fallback to full image if boundary detection fails
            cropped_image = image
            original_size = (image.shape[1], image.shape[0])
            boundary_info = None
    else:
        cropped_image = image
        original_size = (image.shape[1], image.shape[0])
        boundary_info = None
    
    # Load learned positions
    learned_data = load_learned_positions(doc_type)
    if learned_data is None:
        return {
            'is_fake': False,
            'confidence': 0.5,
            'issues': ['no_learned_positions_available'],
            'method': 'position_based',
            'position_details': {}
        }
    
    # Load model if not provided
    if model is None:
        try:
            model = create_enhanced_ensemble_model(
                input_shape=(150, 150, 3),
                num_classes=4,
                predict_positions=True
            )
            model = compile_enhanced_model(model, learning_rate=0.001, predict_positions=True)
            model.load_weights('models/kyc_validator_enhanced.h5')
        except Exception as e:
            return {
                'is_fake': False,
                'confidence': 0.5,
                'issues': [f'model_load_error: {str(e)}'],
                'method': 'position_based',
                'position_details': {}
            }
    
    # Predict positions using model (on cropped image if boundaries detected)
    try:
        # Use cropped image for prediction if boundaries were detected
        prediction_image = cropped_image if boundary_info else image
        predicted_positions = predict_positions(model, prediction_image)
        parsed_positions = parse_positions(predicted_positions)
        
        # Step 2: Normalize positions relative to document boundaries (if detected)
        if boundary_info and boundary_info['confidence'] > 0.5:
            # Normalize each position relative to document boundaries
            normalized_positions = {}
            for element_name, position in parsed_positions.items():
                normalized_pos = normalize_position_to_boundaries(
                    tuple(position),
                    boundary_info['boundaries'],
                    original_size
                )
                normalized_positions[element_name] = np.array(normalized_pos)
            parsed_positions = normalized_positions
    except Exception as e:
        return {
            'is_fake': False,
            'confidence': 0.5,
            'issues': [f'prediction_error: {str(e)}'],
            'method': 'position_based',
            'position_details': {},
            'boundary_detection': boundary_info
        }
    
    # Get learned position statistics
    position_stats = learned_data.get('position_stats', {})
    
    # Validate each element
    element_names = ['photo', 'name', 'dob', 'document_number']
    
    for element_name in element_names:
        if element_name not in parsed_positions:
            continue
        
        if element_name not in position_stats:
            continue
        
        predicted = parsed_positions[element_name]
        learned_mean = np.array(position_stats[element_name]['mean'])
        learned_std = np.array(position_stats[element_name]['std'])
        
        # Calculate deviation
        avg_z_score, is_suspicious = calculate_position_deviation(
            predicted, learned_mean, learned_std, threshold_sigma
        )
        
        # Calculate overlap
        overlap_ratio = calculate_overlap_ratio(predicted, learned_mean)
        
        # Store details
        position_details[element_name] = {
            'predicted': predicted.tolist(),
            'learned_mean': learned_mean.tolist(),
            'learned_std': learned_std.tolist(),
            'z_score_avg': avg_z_score,
            'z_score_max': float(np.max(np.abs((predicted - learned_mean) / (learned_std + 1e-6)))),
            'overlap_ratio': overlap_ratio,
            'is_valid': overlap_ratio >= min_overlap and not is_suspicious
        }
        
        # Flag issues
        if not position_details[element_name]['is_valid']:
            if overlap_ratio < min_overlap:
                issues.append(f'{element_name}_position_mismatch')
                confidence -= 0.2
            if is_suspicious:
                issues.append(f'{element_name}_position_anomaly')
                confidence -= 0.15
    
    # Determine if fake
    is_fake = len(issues) > 0 or confidence < 0.7
    
    result = {
        'is_fake': is_fake,
        'confidence': max(0.0, min(1.0, confidence)),
        'issues': issues,
        'method': 'position_based',
        'position_details': position_details,
        'summary': {
            'total_elements_checked': len(position_details),
            'valid_elements': sum(1 for v in position_details.values() if v['is_valid']),
            'suspicious_elements': sum(1 for v in position_details.values() if not v['is_valid'])
        }
    }
    
    # Add boundary detection info
    if boundary_info:
        result['boundary_detection'] = {
            'detected': True,
            'confidence': boundary_info['confidence'],
            'method': boundary_info['method'],
            'boundaries': boundary_info['boundaries']
        }
    else:
        result['boundary_detection'] = {
            'detected': False,
            'confidence': 0.0,
            'method': 'none'
        }
    
    return result


def comprehensive_position_validation(image_path: str, doc_type: str = 'aadhaar') -> Dict:
    """
    Complete validation pipeline using position-based detection
    
    Args:
        image_path: Path to image file
        doc_type: Document type ('aadhaar' or 'pan')
    
    Returns:
        Complete validation results
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return {
            'error': 'Could not load image',
            'is_fake': False,
            'confidence': 0.0
        }
    
    # Load model once
    try:
        model = create_enhanced_ensemble_model(
            input_shape=(150, 150, 3),
            num_classes=4,
            predict_positions=True
        )
        model = compile_enhanced_model(model, learning_rate=0.001, predict_positions=True)
        model.load_weights('models/kyc_validator_enhanced.h5')
    except Exception as e:
        return {
            'error': f'Could not load model: {str(e)}',
            'is_fake': False,
            'confidence': 0.0
        }
    
    # Run position-based detection
    result = detect_fake_using_positions(image, doc_type, model)
    
    return result


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python position_based_fake_detector.py <image_path> [doc_type]")
        print("Example: python position_based_fake_detector.py test.jpg aadhaar")
        sys.exit(1)
    
    image_path = sys.argv[1]
    doc_type = sys.argv[2] if len(sys.argv) > 2 else 'aadhaar'
    
    print("="*60)
    print("POSITION-BASED FAKE DETECTION")
    print("="*60)
    print(f"\nImage: {image_path}")
    print(f"Document Type: {doc_type}")
    print("\nAnalyzing...")
    
    result = comprehensive_position_validation(image_path, doc_type)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nIs Fake: {result.get('is_fake', False)}")
    print(f"Confidence: {result.get('confidence', 0.0):.2%}")
    print(f"Issues: {result.get('issues', [])}")
    
    if 'position_details' in result:
        print("\nPosition Details:")
        for element, details in result['position_details'].items():
            print(f"\n  {element.upper()}:")
            print(f"    Valid: {details['is_valid']}")
            print(f"    Overlap Ratio: {details['overlap_ratio']:.2%}")
            print(f"    Max Z-Score: {details['z_score_max']:.2f}")
    
    if 'summary' in result:
        print("\nSummary:")
        print(f"  Elements Checked: {result['summary']['total_elements_checked']}")
        print(f"  Valid Elements: {result['summary']['valid_elements']}")
        print(f"  Suspicious Elements: {result['summary']['suspicious_elements']}")

