"""
Verify actual background colors in your dataset
This helps confirm what colors PAN and Aadhaar cards actually have
"""

import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_image_colors(image_path: str) -> dict:
    """Analyze dominant colors in an image"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate histogram for hue channel
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    dominant_hue = int(np.argmax(hist_h))
    
    # Calculate mean saturation and value
    mean_saturation = np.mean(hsv[:, :, 1])
    mean_value = np.mean(hsv[:, :, 2])
    
    # Determine color category
    color_category = "unknown"
    if mean_saturation < 30 and mean_value > 200:
        color_category = "white"
    elif 100 <= dominant_hue <= 130 and mean_saturation > 50:
        color_category = "blue"
    elif 100 <= dominant_hue <= 120:
        color_category = "blue-green"
    elif mean_saturation < 30:
        color_category = "gray/white"
    else:
        color_category = f"other (hue={dominant_hue})"
    
    return {
        'dominant_hue': dominant_hue,
        'mean_saturation': float(mean_saturation),
        'mean_value': float(mean_value),
        'color_category': color_category
    }


def analyze_dataset(data_dir: str = 'data'):
    """Analyze colors in entire dataset"""
    results = {
        'pan': defaultdict(list),
        'aadhaar': defaultdict(list),
        'fake': defaultdict(list),
        'other': defaultdict(list)
    }
    
    # Traverse dataset directory
    for split in ['train', 'val', 'test']:
        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            continue
        
        for doc_type in ['pan', 'aadhaar', 'fake', 'other']:
            doc_dir = split_dir / doc_type
            if not doc_dir.exists():
                continue
            
            print(f"\nAnalyzing {split}/{doc_type}...")
            image_files = list(doc_dir.glob('*.jpg')) + list(doc_dir.glob('*.png'))
            
            for img_path in image_files[:50]:  # Sample first 50 images
                color_info = analyze_image_colors(str(img_path))
                if color_info:
                    results[doc_type]['hues'].append(color_info['dominant_hue'])
                    results[doc_type]['saturations'].append(color_info['mean_saturation'])
                    results[doc_type]['values'].append(color_info['mean_value'])
                    results[doc_type]['categories'].append(color_info['color_category'])
    
    # Print summary
    print("\n" + "="*60)
    print("COLOR ANALYSIS SUMMARY")
    print("="*60)
    
    for doc_type, stats in results.items():
        if not stats['hues']:
            continue
        
        print(f"\n{doc_type.upper()}:")
        print(f"  Total images analyzed: {len(stats['hues'])}")
        print(f"  Average Hue: {np.mean(stats['hues']):.1f}")
        print(f"  Average Saturation: {np.mean(stats['saturations']):.1f}")
        print(f"  Average Value (Brightness): {np.mean(stats['values']):.1f}")
        
        # Count color categories
        from collections import Counter
        category_counts = Counter(stats['categories'])
        print(f"  Color Categories:")
        for cat, count in category_counts.most_common():
            print(f"    - {cat}: {count} ({count/len(stats['categories'])*100:.1f}%)")
        
        # Determine expected color
        most_common = category_counts.most_common(1)[0][0]
        print(f"  → Most common: {most_common}")
    
    print("\n" + "="*60)
    print("EXPECTED COLORS (Based on Analysis):")
    print("="*60)
    print("PAN Card: Should be blue-green")
    print("Aadhaar Card: Should be white (or blue for Baal Aadhaar)")
    print("\nIf your dataset shows different colors, you may need to:")
    print("1. Verify your dataset images are correct")
    print("2. Adjust color detection ranges")
    print("3. Retrain model with correct color understanding")
    
    return results


if __name__ == "__main__":
    print("Analyzing colors in your dataset...")
    print("This will help verify if color assumptions are correct.\n")
    
    results = analyze_dataset()
    
    # Save results
    import json
    with open('color_analysis_results.json', 'w') as f:
        # Convert numpy types to native Python types
        json_results = {}
        for doc_type, stats in results.items():
            json_results[doc_type] = {
                'hues': [float(h) for h in stats['hues']],
                'saturations': [float(s) for s in stats['saturations']],
                'values': [float(v) for v in stats['values']],
                'categories': stats['categories']
            }
        json.dump(json_results, f, indent=2)
    
    print("\n✅ Results saved to 'color_analysis_results.json'")

