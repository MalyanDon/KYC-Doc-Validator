"""
Script to move PAN card images from train/val to test set
"""

import os
import shutil
import random
from pathlib import Path

def move_images_to_test(source_dir, target_dir, num_images):
    """Move random images from source to target directory"""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(source_path.glob('*.jpg')) + list(source_path.glob('*.jpeg')) + list(source_path.glob('*.png'))
    
    if len(image_files) == 0:
        print(f"No images found in {source_dir}")
        return 0
    
    # Randomly select images to move
    num_to_move = min(num_images, len(image_files))
    selected_images = random.sample(image_files, num_to_move)
    
    # Move images
    moved_count = 0
    for img_file in selected_images:
        try:
            target_file = target_path / img_file.name
            shutil.move(str(img_file), str(target_file))
            moved_count += 1
        except Exception as e:
            print(f"Error moving {img_file.name}: {e}")
    
    return moved_count

def main():
    print("=" * 60)
    print("Moving PAN Card Images to Test Set")
    print("=" * 60)
    
    # Paths
    train_pan = "data/train/pan"
    val_pan = "data/val/pan"
    test_pan = "data/test/pan"
    
    # Count current images
    train_count = len(list(Path(train_pan).glob('*.jpg'))) + len(list(Path(train_pan).glob('*.jpeg'))) + len(list(Path(train_pan).glob('*.png')))
    val_count = len(list(Path(val_pan).glob('*.jpg'))) + len(list(Path(val_pan).glob('*.jpeg'))) + len(list(Path(val_pan).glob('*.png')))
    test_count = len(list(Path(test_pan).glob('*.jpg'))) + len(list(Path(test_pan).glob('*.jpeg'))) + len(list(Path(test_pan).glob('*.png')))
    
    print(f"\nCurrent Distribution:")
    print(f"  Train PAN: {train_count} images")
    print(f"  Val PAN:   {val_count} images")
    print(f"  Test PAN:  {test_count} images")
    
    # Target: Move ~250 images to test (similar to Aadhaar test count of 265)
    target_test_count = 250
    
    if test_count >= target_test_count:
        print(f"\nTest set already has {test_count} images. No need to move.")
        return
    
    images_needed = target_test_count - test_count
    
    print(f"\nTarget: {target_test_count} images in test set")
    print(f"Need to move: {images_needed} images")
    
    # Move from train (larger set)
    # Take 70% from train, 30% from val to maintain balance
    from_train = int(images_needed * 0.7)
    from_val = images_needed - from_train
    
    print(f"\nMoving {from_train} images from train...")
    moved_train = move_images_to_test(train_pan, test_pan, from_train)
    
    print(f"Moving {from_val} images from val...")
    moved_val = move_images_to_test(val_pan, test_pan, from_val)
    
    # Final counts
    train_final = len(list(Path(train_pan).glob('*.jpg'))) + len(list(Path(train_pan).glob('*.jpeg'))) + len(list(Path(train_pan).glob('*.png')))
    val_final = len(list(Path(val_pan).glob('*.jpg'))) + len(list(Path(val_pan).glob('*.jpeg'))) + len(list(Path(val_pan).glob('*.png')))
    test_final = len(list(Path(test_pan).glob('*.jpg'))) + len(list(Path(test_pan).glob('*.jpeg'))) + len(list(Path(test_pan).glob('*.png')))
    
    print("\n" + "=" * 60)
    print("Final Distribution:")
    print(f"  Train PAN: {train_final} images (moved {moved_train})")
    print(f"  Val PAN:   {val_final} images (moved {moved_val})")
    print(f"  Test PAN:  {test_final} images (total moved: {moved_train + moved_val})")
    print("=" * 60)
    
    if moved_train + moved_val > 0:
        print("\n[SUCCESS] Successfully moved PAN images to test set!")
    else:
        print("\n[WARNING] No images were moved.")

if __name__ == "__main__":
    main()

