"""
Quick Test Script for KYC Document Validator
Tests the model on a single image or batch of images
"""

import sys
import os
import argparse
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import create_ensemble_model
from ocr_utils import extract_document_info, mock_uidai_validation, mock_it_validation
from fake_detector import comprehensive_fake_detection


def preprocess_image(image: np.ndarray, target_size: tuple = (150, 150)) -> np.ndarray:
    """Preprocess image for model input"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    return image


def predict_document(image_path: str, model_path: str = 'models/kyc_validator.h5', verbose: bool = True):
    """
    Predict document type and authenticity for a single image
    
    Args:
        image_path: Path to image file
        model_path: Path to trained model weights
        verbose: Print detailed results
    """
    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("💡 Train the model first: python src/train.py")
        return None
    
    if verbose:
        print(f"📦 Loading model from {model_path}...")
    
    model = create_ensemble_model(input_shape=(150, 150, 3), num_classes=4)
    model.load_weights(model_path)
    
    # Load image
    if not os.path.exists(image_path):
        print(f"❌ Image not found at {image_path}")
        return None
    
    if verbose:
        print(f"📷 Loading image from {image_path}...")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    preprocessed = preprocess_image(image_rgb)
    input_batch = np.expand_dims(preprocessed, axis=0)
    
    # Predict
    if verbose:
        print("🔮 Running predictions...")
    
    predictions = model.predict(input_batch, verbose=0)
    class_probs = predictions[0][0]
    auth_score = float(predictions[1][0][0])
    
    class_names = ['Aadhaar', 'PAN', 'Fake', 'Other']
    predicted_class_idx = np.argmax(class_probs)
    predicted_class = class_names[predicted_class_idx]
    confidence = float(class_probs[predicted_class_idx])
    
    # OCR
    if verbose:
        print("📝 Extracting text with OCR...")
    
    ocr_info = extract_document_info(image)
    
    # Fake detection
    if verbose:
        print("🔍 Running fake detection...")
    
    fake_result = comprehensive_fake_detection(
        image, 
        predicted_class.lower(), 
        ocr_info['raw_text']
    )
    
    # API validation
    api_validation = {}
    if ocr_info['aadhaar_number']:
        api_validation['aadhaar'] = mock_uidai_validation(ocr_info['aadhaar_number'])
    if ocr_info['pan_number']:
        api_validation['pan'] = mock_it_validation(ocr_info['pan_number'])
    
    # Compile results
    results = {
        'image_path': image_path,
        'prediction': {
            'type': predicted_class,
            'confidence': confidence,
            'all_probabilities': {name: float(prob) for name, prob in zip(class_names, class_probs)}
        },
        'authenticity': {
            'score': auth_score,
            'is_fake': fake_result['is_fake'],
            'issues': fake_result['issues']
        },
        'ocr': {
            'aadhaar_number': ocr_info['aadhaar_number'],
            'pan_number': ocr_info['pan_number'],
            'text_length': ocr_info['text_length']
        },
        'api_validation': api_validation
    }
    
    # Print results
    if verbose:
        print("\n" + "="*60)
        print("📊 PREDICTION RESULTS")
        print("="*60)
        print(f"📄 Document Type: {predicted_class}")
        print(f"🎯 Confidence: {confidence:.2%}")
        print(f"✅ Authenticity Score: {auth_score:.2%}")
        print(f"🚨 Is Fake: {'Yes' if fake_result['is_fake'] else 'No'}")
        
        if fake_result['issues']:
            print(f"\n⚠️  Issues Detected:")
            for issue in fake_result['issues']:
                print(f"   - {issue.replace('_', ' ').title()}")
        else:
            print("\n✅ No issues detected")
        
        print(f"\n📝 Extracted Information:")
        if ocr_info['aadhaar_number']:
            print(f"   Aadhaar: {ocr_info['aadhaar_number']}")
        if ocr_info['pan_number']:
            print(f"   PAN: {ocr_info['pan_number']}")
        print(f"   Text Length: {ocr_info['text_length']} characters")
        
        print(f"\n📊 All Class Probabilities:")
        for name, prob in results['prediction']['all_probabilities'].items():
            bar = "█" * int(prob * 20)
            print(f"   {name:10s}: {prob:6.2%} {bar}")
        
        print("="*60)
    
    return results


def test_batch(image_dir: str, model_path: str = 'models/kyc_validator.h5'):
    """
    Test model on all images in a directory
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"📁 Found {len(image_files)} images in {image_dir}")
    print(f"🧪 Testing all images...\n")
    
    results = []
    for img_path in image_files:
        print(f"\n{'='*60}")
        print(f"Testing: {img_path.name}")
        print('='*60)
        result = predict_document(str(img_path), model_path, verbose=True)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n\n{'='*60}")
    print("📊 BATCH TEST SUMMARY")
    print('='*60)
    print(f"Total Images: {len(results)}")
    
    if results:
        types = [r['prediction']['type'] for r in results]
        from collections import Counter
        type_counts = Counter(types)
        print(f"\nDocument Types:")
        for doc_type, count in type_counts.items():
            print(f"  {doc_type}: {count}")
        
        fake_count = sum(1 for r in results if r['authenticity']['is_fake'])
        print(f"\nFake Documents: {fake_count}/{len(results)}")
        
        avg_confidence = np.mean([r['prediction']['confidence'] for r in results])
        avg_authenticity = np.mean([r['authenticity']['score'] for r in results])
        print(f"\nAverage Confidence: {avg_confidence:.2%}")
        print(f"Average Authenticity: {avg_authenticity:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test KYC Document Validator')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--dir', type=str, help='Path to directory with images')
    parser.add_argument('--model', type=str, default='models/kyc_validator.h5',
                       help='Path to trained model weights')
    
    args = parser.parse_args()
    
    if args.image:
        # Test single image
        predict_document(args.image, args.model, verbose=True)
    elif args.dir:
        # Test batch
        test_batch(args.dir, args.model)
    else:
        print("Usage:")
        print("  Test single image: python test_model.py --image path/to/image.jpg")
        print("  Test directory:   python test_model.py --dir path/to/images/")
        print("  With custom model: python test_model.py --image image.jpg --model models/custom.h5")

