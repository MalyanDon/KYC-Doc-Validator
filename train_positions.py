"""
Script to train position detector from real documents
Learns where elements should be positioned
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.append('src')

from trainable_layout_detector import LayoutPositionLearner


def train_from_images(images_dir: str, doc_type: str, output_file: str):
    """Train position detector from images"""
    print("="*60)
    print("TRAINING POSITION DETECTOR FROM IMAGES")
    print("="*60)
    print(f"Document Type: {doc_type.upper()}")
    print(f"Images Directory: {images_dir}")
    print(f"Output File: {output_file}")
    print("="*60)
    
    if not os.path.exists(images_dir):
        print(f"[ERROR] Directory {images_dir} not found")
        return
    
    # Count images
    images = list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.png'))
    print(f"\n[INFO] Found {len(images)} images")
    
    if len(images) == 0:
        print("[ERROR] No images found. Please add images to the directory.")
        return
    
    # Create learner
    learner = LayoutPositionLearner(doc_type=doc_type)
    
    # Learn positions
    print("\n[INFO] Learning positions from images...")
    learned = learner.learn_from_images(images_dir, auto_annotate=True)
    
    if not learned:
        print("[ERROR] Failed to learn positions")
        return
    
    # Save learned positions
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    learner.save_learned_positions(output_file)
    
    # Display results
    print("\n" + "="*60)
    print("LEARNED POSITIONS")
    print("="*60)
    
    if 'photo_region' in learned:
        photo = learned['photo_region']
        print(f"\n[PHOTO] Photo Region:")
        print(f"   Normalized: ({photo[0]:.3f}, {photo[1]:.3f}, {photo[2]:.3f}, {photo[3]:.3f})")
        if 'photo' in learner.position_stats:
            stats = learner.position_stats['photo']
            print(f"   Learned from: {stats['count']} samples")
            print(f"   Std Dev: ({stats['std'][0]:.3f}, {stats['std'][1]:.3f}, {stats['std'][2]:.3f}, {stats['std'][3]:.3f})")
    
    if 'text_regions' in learned:
        print(f"\n📝 Text Regions:")
        for label, pos in learned['text_regions'].items():
            print(f"   {label}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}, {pos[3]:.3f})")
            if label in learner.position_stats:
                stats = learner.position_stats[label]
                print(f"      Learned from: {stats['count']} samples")
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print(f"✅ Learned positions saved to: {output_file}")
    print("\n💡 These positions will be used automatically by layout_validator.py")
    print("="*60)


def train_from_annotations(annotations_dir: str, doc_type: str, output_file: str):
    """Train position detector from annotations"""
    print("="*60)
    print("TRAINING POSITION DETECTOR FROM ANNOTATIONS")
    print("="*60)
    print(f"Document Type: {doc_type.upper()}")
    print(f"Annotations Directory: {annotations_dir}")
    print(f"Output File: {output_file}")
    print("="*60)
    
    if not os.path.exists(annotations_dir):
        print(f"[ERROR] Directory {annotations_dir} not found")
        return
    
    # Count annotations
    annotations = list(Path(annotations_dir).glob('*.json'))
    print(f"\n[INFO] Found {len(annotations)} annotation files")
    
    if len(annotations) == 0:
        print("[ERROR] No annotations found.")
        print("[INFO] Create annotations using: python src/annotation_helper.py")
        return
    
    # Create learner
    learner = LayoutPositionLearner(doc_type=doc_type)
    
    # Learn positions
    print("\n[INFO] Learning positions from annotations...")
    learned = learner.learn_from_annotations(annotations_dir)
    
    if not learned:
        print("[ERROR] Failed to learn positions")
        return
    
    # Save learned positions
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    learner.save_learned_positions(output_file)
    
    # Display results
    print("\n" + "="*60)
    print("LEARNED POSITIONS")
    print("="*60)
    print(f"\n📸 Photo: {learned.get('photo_region')}")
    print(f"📝 Text Regions: {learned.get('text_regions')}")
    print("\n✅ Training complete!")
    print(f"✅ Learned positions saved to: {output_file}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train position detector from real documents')
    parser.add_argument('--method', type=str, choices=['images', 'annotations'], default='images',
                       help='Training method: images (auto-detect) or annotations (manual)')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory with images or annotations')
    parser.add_argument('--doc_type', type=str, choices=['aadhaar', 'pan'], default='aadhaar',
                       help='Document type')
    parser.add_argument('--output', type=str, 
                       default='models/learned_{doc_type}_positions.json',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Format output path
    output_file = args.output.format(doc_type=args.doc_type)
    
    if args.method == 'images':
        train_from_images(args.input_dir, args.doc_type, output_file)
    else:
        train_from_annotations(args.input_dir, args.doc_type, output_file)

