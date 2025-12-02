#!/usr/bin/env python3
"""
Recovery script to save the model from training history.
This script reconstructs the model architecture and loads weights if available.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import create_ensemble_model, compile_model

def main():
    print("🔧 Model Recovery Script")
    print("=" * 60)
    
    # Model configuration
    input_shape = (150, 150, 3)
    num_classes = 4
    model_save_path = 'models/kyc_validator.h5'
    
    print(f"\n1. Creating model architecture...")
    model = create_ensemble_model(
        input_shape=input_shape,
        num_classes=num_classes,
        fine_tune_layers=4,
        use_flatten_vgg=False
    )
    
    print(f"2. Compiling model...")
    compile_model(model, learning_rate=0.001)
    
    print(f"3. Model architecture created successfully!")
    print(f"   Total parameters: {model.count_params():,}")
    
    # Try to load weights if they exist
    weights_path = model_save_path.replace('.h5', '_weights.h5')
    if os.path.exists(weights_path):
        print(f"\n4. Loading weights from {weights_path}...")
        try:
            model.load_weights(weights_path)
            print("   ✅ Weights loaded successfully!")
        except Exception as e:
            print(f"   ⚠️  Could not load weights: {e}")
            print("   Saving model architecture only...")
    else:
        print(f"\n4. No weights file found at {weights_path}")
        print("   Saving model architecture only (untrained model)...")
    
    # Save the model
    print(f"\n5. Saving model to {model_save_path}...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    if os.path.exists(model_save_path):
        os.remove(model_save_path)
    
    try:
        model.save(model_save_path)
        print(f"   ✅ Model saved successfully to {model_save_path}!")
        print(f"   File size: {os.path.getsize(model_save_path) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"   ❌ Error saving model: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ Recovery complete!")
    return 0

if __name__ == '__main__':
    sys.exit(main())

