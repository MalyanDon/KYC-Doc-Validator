"""
Script to create validation set from training data
Splits train data into train/val (typically 85/15 or 80/20)
"""

import os
import shutil
import random
from pathlib import Path


def create_validation_set(data_dir='data', split_ratio=0.15, seed=42):
    """
    Create validation set by splitting training data
    
    Args:
        data_dir: Path to data directory
        split_ratio: Fraction of train data to move to val (default: 0.15 = 15%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    print("="*60)
    print("CREATING VALIDATION SET")
    print("="*60)
    print(f"Split ratio: {split_ratio*100:.0f}% to validation")
    print("="*60)
    
    train_dir = Path(data_dir) / 'train'
    val_dir = Path(data_dir) / 'val'
    
    if not train_dir.exists():
        print(f"❌ Error: {train_dir} not found")
        return
    
    # Process each class
    classes = ['aadhaar', 'pan', 'fake', 'other']
    total_moved = 0
    
    for class_name in classes:
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        
        if not train_class_dir.exists():
            print(f"\n⚠️  {class_name}: No train directory found, skipping")
            continue
        
        # Find all images
        images = list(train_class_dir.glob('*.jpg')) + \
                 list(train_class_dir.glob('*.jpeg')) + \
                 list(train_class_dir.glob('*.png')) + \
                 list(train_class_dir.glob('*.JPG')) + \
                 list(train_class_dir.glob('*.JPEG')) + \
                 list(train_class_dir.glob('*.PNG'))
        
        if len(images) == 0:
            print(f"\n⚠️  {class_name}: No images found, skipping")
            continue
        
        # Shuffle
        random.shuffle(images)
        
        # Calculate how many to move
        val_count = int(len(images) * split_ratio)
        
        if val_count == 0:
            print(f"\n⚠️  {class_name}: Not enough images ({len(images)}) to split")
            continue
        
        # Create val directory
        val_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Move images
        moved = 0
        for img in images[:val_count]:
            try:
                shutil.move(str(img), str(val_class_dir / img.name))
                moved += 1
            except Exception as e:
                print(f"⚠️  Error moving {img.name}: {e}")
        
        total_moved += moved
        
        print(f"\n✅ {class_name}:")
        print(f"   Train: {len(images)} → {len(images) - moved} images")
        print(f"   Val:   0 → {moved} images")
        print(f"   Moved: {moved} images")
    
    print("\n" + "="*60)
    print(f"✅ Total images moved to validation: {total_moved}")
    print("="*60)
    
    return total_moved


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create validation set from training data')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--split_ratio', type=float, default=0.15,
                       help='Fraction of train data to move to val (default: 0.15 = 15%%)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("\n📚 Creating validation set...")
    print(f"   Data directory: {args.data_dir}")
    print(f"   Split ratio: {args.split_ratio*100:.0f}%")
    print(f"   Random seed: {args.seed}")
    print()
    
    create_validation_set(args.data_dir, args.split_ratio, args.seed)
    
    print("\n💡 Next step: Verify dataset")
    print("   python prepare_dataset.py --count --verify")

