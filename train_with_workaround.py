#!/usr/bin/env python3
"""
Training script with macOS TensorFlow workaround
"""

import os
import sys

# macOS TensorFlow workaround
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Try to work around the mutex issue
try:
    import tensorflow as tf
    # Set memory growth to avoid issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU config warning: {e}")
except Exception as e:
    print(f"TensorFlow initialization warning: {e}")

# Now import and run training
if __name__ == "__main__":
    # Import after setting environment
    sys.path.insert(0, 'src')
    
    # Run training
    import argparse
    from train import train_model
    
    parser = argparse.ArgumentParser(description='Train KYC Document Validator')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--model_save_path', type=str, default='models/kyc_validator.h5',
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    print("="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("="*60)
    print()
    
    try:
        train_model(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            model_save_path=args.model_save_path
        )
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

