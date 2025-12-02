"""
Test Position-Based Fake Detection
Demonstrates how to use position predictions to detect fake Aadhaar cards
"""

import cv2
import sys
import os
sys.path.append('src')

from position_based_fake_detector import comprehensive_position_validation

def test_position_detection(image_path: str, doc_type: str = 'aadhaar'):
    """
    Test position-based fake detection on an image
    
    Args:
        image_path: Path to the image file
        doc_type: Document type ('aadhaar' or 'pan')
    """
    print("="*70)
    print("POSITION-BASED FAKE DETECTION TEST")
    print("="*70)
    print(f"\nImage: {image_path}")
    print(f"Document Type: {doc_type.upper()}")
    
    if not os.path.exists(image_path):
        print(f"\n[ERROR] Image not found: {image_path}")
        return
    
    print("\n[INFO] Loading model and analyzing positions...")
    print("   - Predicting element positions using trained model")
    print("   - Comparing against learned positions from real documents")
    print("   - Calculating deviations and overlap ratios")
    
    # Run detection
    result = comprehensive_position_validation(image_path, doc_type)
    
    # Display results
    print("\n" + "="*70)
    print("DETECTION RESULTS")
    print("="*70)
    
    if 'error' in result:
        print(f"\n[ERROR] {result['error']}")
        return
    
    print(f"\n[RESULT] Is Fake: {'YES' if result['is_fake'] else 'NO'}")
    print(f"[CONFIDENCE] {result['confidence']:.1%}")
    
    if result.get('issues'):
        print(f"\n[ISSUES DETECTED]")
        for issue in result['issues']:
            print(f"   - {issue}")
    else:
        print("\n[STATUS] No issues detected - positions match expected layout")
    
    # Detailed position analysis
    if 'position_details' in result and result['position_details']:
        print("\n" + "-"*70)
        print("POSITION ANALYSIS DETAILS")
        print("-"*70)
        
        for element_name, details in result['position_details'].items():
            status = "VALID" if details['is_valid'] else "SUSPICIOUS"
            status_symbol = "[OK]" if details['is_valid'] else "[FAKE]"
            
            print(f"\n{element_name.upper()} {status_symbol}")
            print(f"   Status: {status}")
            print(f"   Overlap with learned positions: {details['overlap_ratio']:.1%}")
            print(f"   Deviation (Z-score): {details['z_score_max']:.2f} standard deviations")
            
            if not details['is_valid']:
                print(f"   [WARNING] Position does not match expected layout!")
                if details['overlap_ratio'] < 0.5:
                    print(f"   [REASON] Low overlap ({details['overlap_ratio']:.1%}) - element in wrong location")
                if details['z_score_max'] > 2.0:
                    print(f"   [REASON] High deviation ({details['z_score_max']:.2f}σ) - unusual position")
    
    # Summary
    if 'summary' in result:
        print("\n" + "-"*70)
        print("SUMMARY")
        print("-"*70)
        print(f"   Elements Checked: {result['summary']['total_elements_checked']}")
        print(f"   Valid Elements: {result['summary']['valid_elements']}")
        print(f"   Suspicious Elements: {result['summary']['suspicious_elements']}")
    
    print("\n" + "="*70)
    print("HOW IT WORKS")
    print("="*70)
    print("""
1. Model predicts positions of:
   - Photo region
   - Name text region
   - Date of Birth text region
   - Document number region

2. Compares predicted positions against learned positions from:
   - 1,416 real Aadhaar cards (for photo)
   - 712 real documents (for name)
   - 708 real documents (for DOB)
   - 140 real documents (for document number)

3. Flags as FAKE if:
   - Positions deviate >2 standard deviations from learned positions
   - Overlap with expected positions <50%
   - Elements are in wrong locations (e.g., photo on wrong side)

4. This detects:
   - Tampered documents (elements moved)
   - Fake documents (wrong layout)
   - Scanned/photocopied fakes (position errors)
    """)
    
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_position_detection.py <image_path> [doc_type]")
        print("\nExample:")
        print("  python test_position_detection.py data/test/aadhaar/sample.jpg aadhaar")
        print("  python test_position_detection.py data/test/pan/sample.jpg pan")
        sys.exit(1)
    
    image_path = sys.argv[1]
    doc_type = sys.argv[2] if len(sys.argv) > 2 else 'aadhaar'
    
    test_position_detection(image_path, doc_type)

