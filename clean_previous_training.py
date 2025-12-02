"""
Clean all previous training outputs and models
"""

import os
import glob
from pathlib import Path

def clean_training_outputs():
    """Remove all previous training outputs"""
    print("=" * 60)
    print("Cleaning Previous Training Outputs")
    print("=" * 60)
    
    files_to_remove = []
    
    # Model files
    model_patterns = [
        "models/*.h5",
        "models/*.hdf5",
        "models/*.pb",
        "models/*.pkl",
        "*.h5",
        "*.hdf5"
    ]
    
    # Training outputs
    output_patterns = [
        "confusion_matrix.png",
        "training_history.png",
        "*.png",  # Other plots
        "training_log.txt",
        "logs/**/*",
        "tensorboard_logs/**/*"
    ]
    
    # Collect files
    for pattern in model_patterns + output_patterns:
        files = glob.glob(pattern, recursive=True)
        files_to_remove.extend(files)
    
    # Remove files
    removed_count = 0
    for file_path in files_to_remove:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                removed_count += 1
                print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
    
    # Keep position JSON files (they're useful)
    print(f"\n[INFO] Keeping learned position JSON files (learned_aadhaar_positions.json, learned_pan_positions.json)")
    
    print("\n" + "=" * 60)
    print(f"Cleanup Complete: Removed {removed_count} files")
    print("=" * 60)

if __name__ == "__main__":
    clean_training_outputs()

