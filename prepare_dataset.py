"""
Dataset Preparation Helper Script
Helps organize and verify your dataset structure
"""

import os
import shutil
from pathlib import Path
from collections import Counter


def create_dataset_structure(base_dir='data'):
    """Create the required dataset directory structure"""
    splits = ['train', 'val', 'test']
    classes = ['aadhaar', 'pan', 'fake', 'other']
    
    for split in splits:
        for class_name in classes:
            dir_path = os.path.join(base_dir, split, class_name)
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ Created: {dir_path}")
    
    print(f"\n📁 Dataset structure created in {base_dir}/")


def count_images(data_dir='data'):
    """Count images in each folder"""
    splits = ['train', 'val', 'test']
    classes = ['aadhaar', 'pan', 'fake', 'other']
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    print("\n📊 Dataset Statistics:")
    print("="*60)
    
    total_images = 0
    for split in splits:
        split_total = 0
        print(f"\n{split.upper()}:")
        for class_name in classes:
            class_dir = os.path.join(data_dir, split, class_name)
            if os.path.exists(class_dir):
                count = sum(1 for f in os.listdir(class_dir) 
                           if any(f.endswith(ext) for ext in image_extensions))
                split_total += count
                print(f"  {class_name:10s}: {count:4d} images")
            else:
                print(f"  {class_name:10s}: 0 images (directory not found)")
        print(f"  {'TOTAL':10s}: {split_total:4d} images")
        total_images += split_total
    
    print("\n" + "="*60)
    print(f"📦 Total Images: {total_images}")
    print("="*60)
    
    return total_images


def verify_dataset(data_dir='data', min_images_per_class=10):
    """Verify dataset meets minimum requirements"""
    splits = ['train', 'val', 'test']
    classes = ['aadhaar', 'pan', 'fake', 'other']
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    print("\n🔍 Verifying Dataset...")
    print("="*60)
    
    issues = []
    warnings = []
    
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            issues.append(f"❌ {split} directory not found")
            continue
        
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                issues.append(f"❌ {split}/{class_name} directory not found")
                continue
            
            images = [f for f in os.listdir(class_dir) 
                     if any(f.endswith(ext) for ext in image_extensions)]
            count = len(images)
            
            if count == 0:
                warnings.append(f"⚠️  {split}/{class_name}: No images found")
            elif count < min_images_per_class:
                warnings.append(f"⚠️  {split}/{class_name}: Only {count} images (recommended: {min_images_per_class}+)")
            else:
                print(f"✅ {split}/{class_name}: {count} images")
    
    print("\n" + "="*60)
    
    if issues:
        print("\n❌ CRITICAL ISSUES:")
        for issue in issues:
            print(f"  {issue}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  {warning}")
    
    if not issues and not warnings:
        print("\n✅ Dataset verification passed!")
        return True
    elif not issues:
        print("\n⚠️  Dataset has warnings but can be used for training")
        return True
    else:
        print("\n❌ Dataset has critical issues. Please fix before training.")
        return False


def split_dataset(source_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split images from source directory into train/val/test
    
    Usage:
        python prepare_dataset.py --split --source_dir path/to/images --class_name aadhaar
    """
    import random
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        print("❌ Ratios must sum to 1.0")
        return
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    images = [f for f in os.listdir(source_dir) 
             if any(f.endswith(ext) for ext in image_extensions)]
    
    if not images:
        print(f"❌ No images found in {source_dir}")
        return
    
    random.shuffle(images)
    total = len(images)
    
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    
    print(f"📊 Splitting {total} images:")
    print(f"  Train: {len(train_images)} ({len(train_images)/total:.1%})")
    print(f"  Val:   {len(val_images)} ({len(val_images)/total:.1%})")
    print(f"  Test:  {len(test_images)} ({len(test_images)/total:.1%})")
    
    return train_images, val_images, test_images


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset Preparation Helper')
    parser.add_argument('--create', action='store_true', 
                       help='Create dataset directory structure')
    parser.add_argument('--count', action='store_true',
                       help='Count images in dataset')
    parser.add_argument('--verify', action='store_true',
                       help='Verify dataset structure')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    
    args = parser.parse_args()
    
    if args.create:
        create_dataset_structure(args.data_dir)
    
    if args.count:
        count_images(args.data_dir)
    
    if args.verify:
        verify_dataset(args.data_dir)
    
    if not any([args.create, args.count, args.verify]):
        print("Dataset Preparation Helper")
        print("="*60)
        print("\nUsage:")
        print("  Create structure:  python prepare_dataset.py --create")
        print("  Count images:      python prepare_dataset.py --count")
        print("  Verify dataset:    python prepare_dataset.py --verify")
        print("\nOr run all:")
        print("  python prepare_dataset.py --create --count --verify")

