"""
Visualization Tool for Position-Based Validation
Helps visualize expected vs actual positions
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from layout_validator import (
    detect_photo_region, 
    validate_photo_position,
    extract_text_with_positions,
    AADHAAR_LAYOUT,
    PAN_LAYOUT
)


def draw_expected_regions(image: np.ndarray, doc_type: str) -> np.ndarray:
    """
    Draw expected regions on image for visualization
    """
    vis_image = image.copy()
    h, w = image.shape[:2]
    
    if doc_type.lower() == 'aadhaar':
        layout = AADHAAR_LAYOUT
    elif doc_type.lower() == 'pan':
        layout = PAN_LAYOUT
    else:
        return vis_image
    
    # Draw expected photo region
    x_min, y_min, x_max, y_max = layout.photo_region
    px_min = int(x_min * w)
    py_min = int(y_min * h)
    px_max = int(x_max * w)
    py_max = int(y_max * h)
    
    cv2.rectangle(vis_image, (px_min, py_min), (px_max, py_max), (0, 255, 0), 2)
    cv2.putText(vis_image, 'Expected Photo', (px_min, py_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw expected text regions
    for label, tx_min, ty_min, tx_max, ty_max in layout.text_regions:
        tpx_min = int(tx_min * w)
        tpy_min = int(ty_min * h)
        tpx_max = int(tx_max * w)
        tpy_max = int(ty_max * h)
        
        cv2.rectangle(vis_image, (tpx_min, tpy_min), (tpx_max, tpy_max), (255, 255, 0), 1)
        cv2.putText(vis_image, label, (tpx_min, tpy_min + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    return vis_image


def draw_actual_positions(image: np.ndarray, doc_type: str) -> np.ndarray:
    """
    Draw actual detected positions on image
    """
    vis_image = image.copy()
    h, w = image.shape[:2]
    
    # Detect photo
    photo_region = detect_photo_region(image)
    
    if photo_region:
        x, y, pw, ph = photo_region
        cv2.rectangle(vis_image, (x, y), (x + pw, y + ph), (0, 0, 255), 2)
        cv2.putText(vis_image, 'Detected Photo', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Extract text with positions
    texts = extract_text_with_positions(image)
    
    # Draw text positions
    for text_item in texts:
        tx = text_item['x']
        ty = text_item['y']
        tw = text_item['width']
        th = text_item['height']
        text = text_item['text']
        
        cv2.rectangle(vis_image, (tx, ty), (tx + tw, ty + th), (255, 0, 0), 1)
        if len(text) < 20:  # Only show short text labels
            cv2.putText(vis_image, text[:10], (tx, ty - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    
    return vis_image


def visualize_position_comparison(image: np.ndarray, doc_type: str, save_path: Optional[str] = None) -> np.ndarray:
    """
    Create side-by-side comparison of expected vs actual positions
    """
    h, w = image.shape[:2]
    
    # Create comparison image
    comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
    
    # Left side: Expected positions
    expected_vis = draw_expected_regions(image, doc_type)
    comparison[:, :w] = expected_vis
    
    # Add label
    cv2.putText(comparison, 'Expected Positions', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Right side: Actual positions
    actual_vis = draw_actual_positions(image, doc_type)
    comparison[:, w:] = actual_vis
    
    # Add label
    cv2.putText(comparison, 'Actual Positions', (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add legend
    legend_y = h - 100
    cv2.rectangle(comparison, (10, legend_y), (200, legend_y + 80), (0, 0, 0), -1)
    cv2.putText(comparison, 'Green: Expected', (15, legend_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(comparison, 'Red: Detected Photo', (15, legend_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(comparison, 'Blue: Detected Text', (15, legend_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(comparison, 'Yellow: Text Regions', (15, legend_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    if save_path:
        cv2.imwrite(save_path, comparison)
    
    return comparison


def print_position_analysis(image: np.ndarray, doc_type: str):
    """
    Print detailed position analysis
    """
    h, w = image.shape[:2]
    print("\n" + "="*60)
    print("POSITION ANALYSIS")
    print("="*60)
    print(f"Image Size: {w} x {h} pixels")
    print(f"Document Type: {doc_type.upper()}")
    
    # Get layout
    if doc_type.lower() == 'aadhaar':
        layout = AADHAAR_LAYOUT
    elif doc_type.lower() == 'pan':
        layout = PAN_LAYOUT
    else:
        print("Unknown document type")
        return
    
    # Expected photo position
    print("\n📸 EXPECTED PHOTO POSITION:")
    x_min, y_min, x_max, y_max = layout.photo_region
    print(f"  Normalized: ({x_min:.2f}, {y_min:.2f}, {x_max:.2f}, {y_max:.2f})")
    print(f"  Pixels: ({int(x_min*w)}, {int(y_min*h)}) to ({int(x_max*w)}, {int(y_max*h)})")
    print(f"  Size: {int((x_max-x_min)*w)} x {int((y_max-y_min)*h)} pixels")
    
    # Detected photo position
    photo_region = detect_photo_region(image)
    if photo_region:
        x, y, pw, ph = photo_region
        print("\n📷 DETECTED PHOTO POSITION:")
        print(f"  Pixels: ({x}, {y}) to ({x+pw}, {y+ph})")
        print(f"  Size: {pw} x {ph} pixels")
        
        # Normalize
        nx_min = x / w
        ny_min = y / h
        nx_max = (x + pw) / w
        ny_max = (y + ph) / h
        print(f"  Normalized: ({nx_min:.2f}, {ny_min:.2f}, {nx_max:.2f}, {ny_max:.2f})")
        
        # Validate
        result = validate_photo_position(image, doc_type, photo_region)
        print(f"\n✅ VALIDATION:")
        print(f"  Position Valid: {result['position_valid']}")
        print(f"  Overlap Ratio: {result.get('overlap_ratio', 0):.2%}")
        if result['issues']:
            print(f"  Issues: {result['issues']}")
    else:
        print("\n❌ NO PHOTO DETECTED")
    
    # Text positions
    print("\n📝 EXPECTED TEXT REGIONS:")
    for label, tx_min, ty_min, tx_max, ty_max in layout.text_regions:
        print(f"  {label}:")
        print(f"    Normalized: ({tx_min:.2f}, {ty_min:.2f}, {tx_max:.2f}, {ty_max:.2f})")
        print(f"    Pixels: ({int(tx_min*w)}, {int(ty_min*h)}) to ({int(tx_max*w)}, {int(ty_max*h)})")
    
    # Detected text
    texts = extract_text_with_positions(image)
    print(f"\n📄 DETECTED TEXT ({len(texts)} items):")
    for i, text_item in enumerate(texts[:10]):  # Show first 10
        print(f"  {i+1}. '{text_item['text'][:30]}'")
        print(f"     Position: ({text_item['x']}, {text_item['y']})")
        print(f"     Size: {text_item['width']} x {text_item['height']}")
        print(f"     Confidence: {text_item['confidence']}%")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_positions.py <image_path> [doc_type]")
        print("Example: python visualize_positions.py document.jpg aadhaar")
        sys.exit(1)
    
    image_path = sys.argv[1]
    doc_type = sys.argv[2] if len(sys.argv) > 2 else 'aadhaar'
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        sys.exit(1)
    
    # Print analysis
    print_position_analysis(image, doc_type)
    
    # Create visualization
    comparison = visualize_position_comparison(image, doc_type, 'position_comparison.jpg')
    print("\n✅ Visualization saved to: position_comparison.jpg")
    print("   - Left: Expected positions (green)")
    print("   - Right: Actual positions (red=photo, blue=text)")

