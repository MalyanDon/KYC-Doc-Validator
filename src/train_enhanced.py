"""
Enhanced Training Script with Position Prediction
Trains model for: Classification + Authenticity + Position Prediction
"""

import os
import sys
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Tuple, List, Dict
import argparse

# TensorFlow setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import multiprocessing
num_cores = multiprocessing.cpu_count()
os.environ['TF_NUM_INTEROP_THREADS'] = str(num_cores)
os.environ['TF_NUM_INTRAOP_THREADS'] = str(num_cores)
os.environ['OMP_NUM_THREADS'] = str(num_cores)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Import modules
from models_enhanced import create_enhanced_ensemble_model, compile_enhanced_model
from trainable_layout_detector import LayoutPositionLearner
import pytesseract


def load_and_preprocess_image(image_path: str, target_size: tuple = (150, 150)) -> np.ndarray:
    """Load and preprocess image"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    
    return image


def extract_positions_from_image(image_path: str, doc_type: str, 
                                 learned_positions: Dict, use_learned_only: bool = True) -> np.ndarray:
    """
    Extract positions from image - uses learned positions directly (fast)
    If use_learned_only=False, tries OCR/face detection (slow but more accurate)
    Returns normalized positions: [photo(4), name(4), dob(4), number(4)] = 16 values
    """
    positions = np.zeros(16, dtype=np.float32)
    
    # Use learned positions directly (much faster!)
    photo_pos = learned_positions.get('photo_region', [0.1, 0.2, 0.3, 0.4])
    text_regions = learned_positions.get('text_regions', {})
    
    # Fill positions with learned values (fast path)
    if use_learned_only or len(photo_pos) >= 4:
        positions[0:4] = photo_pos[:4]
        if 'name' in text_regions:
            positions[4:8] = text_regions['name'][:4]
        if 'dob' in text_regions:
            positions[8:12] = text_regions['dob'][:4]
        if 'document_number' in text_regions:
            positions[12:16] = text_regions['document_number'][:4]
        return positions
    
    # Slow path: Try to detect actual positions (only if use_learned_only=False)
    image = cv2.imread(image_path)
    if image is None:
        return positions
    
    h, w = image.shape[:2]
    
    try:
        # Detect photo using face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, fw, fh = largest_face
            # Normalize
            positions[0] = max(0, min(1, x / w))  # x_min
            positions[1] = max(0, min(1, y / h))  # y_min
            positions[2] = max(0, min(1, (x + fw) / w))  # x_max
            positions[3] = max(0, min(1, (y + fh) / h))  # y_max
        else:
            # Use learned position as default
            positions[0:4] = photo_pos[:4]
        
        # Extract text positions using OCR
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        n_boxes = len(ocr_data['text'])
        
        # Group text by regions (simplified)
        text_boxes = []
        for i in range(n_boxes):
            if int(ocr_data['conf'][i]) > 60:
                text = ocr_data['text'][i].strip()
                if text and len(text) > 2:
                    x, y, w_box, h_box = ocr_data['left'][i], ocr_data['top'][i], \
                                        ocr_data['width'][i], ocr_data['height'][i]
                    text_boxes.append({
                        'text': text,
                        'x': x / w, 'y': y / h,
                        'x_max': (x + w_box) / w, 'y_max': (y + h_box) / h
                    })
        
        # Assign to name, dob, number based on position and content
        if text_boxes:
            # Sort by y position (top to bottom)
            text_boxes.sort(key=lambda b: b['y'])
            
            # Name is typically first text region
            if len(text_boxes) > 0:
                name_box = text_boxes[0]
                positions[4:8] = [name_box['x'], name_box['y'], name_box['x_max'], name_box['y_max']]
            
            # DOB typically has date pattern
            for box in text_boxes:
                if any(char.isdigit() for char in box['text']) and '/' in box['text']:
                    positions[8:12] = [box['x'], box['y'], box['x_max'], box['y_max']]
                    break
            
            # Number is typically last or has many digits
            for box in reversed(text_boxes):
                if sum(c.isdigit() for c in box['text']) >= 4:
                    positions[12:16] = [box['x'], box['y'], box['x_max'], box['y_max']]
                    break
        
        # Fill missing positions with learned defaults
        if positions[4] == 0 and 'name' in text_regions:
            positions[4:8] = text_regions['name'][:4]
        if positions[8] == 0 and 'dob' in text_regions:
            positions[8:12] = text_regions['dob'][:4]
        if positions[12] == 0 and 'document_number' in text_regions:
            positions[12:16] = text_regions['document_number'][:4]
            
    except Exception as e:
        # If detection fails, use learned positions
        positions[0:4] = photo_pos[:4]
        if 'name' in text_regions:
            positions[4:8] = text_regions['name'][:4]
        if 'dob' in text_regions:
            positions[8:12] = text_regions['dob'][:4]
        if 'document_number' in text_regions:
            positions[12:16] = text_regions['document_number'][:4]
    
    return positions


def load_dataset_with_positions(data_dir: str, target_size: tuple = (150, 150)) -> tuple:
    """
    Load dataset with position labels
    Returns: (X, y_class, y_auth, y_positions)
    """
    X_train, y_train_class, y_train_auth, y_train_pos = [], [], [], []
    X_val, y_val_class, y_val_auth, y_val_pos = [], [], [], []
    X_test, y_test_class, y_test_auth, y_test_pos = [], [], [], []
    
    class_mapping = {'aadhaar': 0, 'pan': 1, 'fake': 2, 'other': 3}
    
    # Load learned positions
    learned_aadhaar = {}
    learned_pan = {}
    
    if os.path.exists('models/learned_aadhaar_positions.json'):
        with open('models/learned_aadhaar_positions.json', 'r') as f:
            data = json.load(f)
            learned_aadhaar = {
                'photo_region': data.get('photo_region', {}).get('mean', [0.1, 0.2, 0.3, 0.4]),
                'text_regions': {k: v.get('mean', [0, 0, 0, 0]) for k, v in data.get('text_regions', {}).items()}
            }
    
    if os.path.exists('models/learned_pan_positions.json'):
        with open('models/learned_pan_positions.json', 'r') as f:
            data = json.load(f)
            learned_pan = {
                'photo_region': data.get('photo_region', {}).get('mean', [0.1, 0.2, 0.3, 0.4]),
                'text_regions': {k: v.get('mean', [0, 0, 0, 0]) for k, v in data.get('text_regions', {}).items()}
            }
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        for class_name, class_idx in class_mapping.items():
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Get learned positions for this document type
            learned_pos = learned_aadhaar if class_name == 'aadhaar' else learned_pan
            
            # Progress indicator
            total_files = len(image_files)
            if total_files > 0:
                print(f"   Loading {class_name} ({split}): {total_files} images...", end='', flush=True)
            
            for idx, img_file in enumerate(image_files):
                if idx > 0 and idx % 100 == 0:
                    print(f".", end='', flush=True)
                
                img_path = os.path.join(class_dir, img_file)
                try:
                    image = load_and_preprocess_image(img_path, target_size)
                    
                    # One-hot encode class
                    class_onehot = keras.utils.to_categorical(class_idx, num_classes=4)
                    
                    # Authenticity: 1 for real, 0 for fake
                    auth_label = 1.0 if class_name != 'fake' else 0.0
                    
                    # Extract positions (use learned positions directly - fast!)
                    positions = extract_positions_from_image(img_path, class_name, learned_pos, use_learned_only=True)
                    
                    if split == 'train':
                        X_train.append(image)
                        y_train_class.append(class_onehot)
                        y_train_auth.append(auth_label)
                        y_train_pos.append(positions)
                    elif split == 'val':
                        X_val.append(image)
                        y_val_class.append(class_onehot)
                        y_val_auth.append(auth_label)
                        y_val_pos.append(positions)
                    else:
                        X_test.append(image)
                        y_test_class.append(class_onehot)
                        y_test_auth.append(auth_label)
                        y_test_pos.append(positions)
                except Exception as e:
                    if idx < 5:  # Only print first few errors
                        print(f"\nWarning: Could not process {img_path}: {e}")
                    continue
            
            if total_files > 0:
                print(" [OK]")
    
    return (
        (np.array(X_train), np.array(y_train_class), np.array(y_train_auth), np.array(y_train_pos)),
        (np.array(X_val), np.array(y_val_class), np.array(y_val_auth), np.array(y_val_pos)),
        (np.array(X_test), np.array(y_test_class), np.array(y_test_auth), np.array(y_test_pos))
    )


class EnhancedDataGenerator(keras.utils.Sequence):
    """Custom data generator for enhanced model with positions"""
    def __init__(self, X_set, y_class_set, y_auth_set, y_pos_set, batch_size, augment=False, target_size=(150, 150)):
        self.X, self.y_class, self.y_auth, self.y_pos = X_set, y_class_set, y_auth_set, y_pos_set
        self.batch_size = batch_size
        self.augment = augment
        self.target_size = target_size
        self.indices = np.arange(len(self.X))
        if self.augment:
            self.datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_X = self.X[indices]
        batch_y_class = self.y_class[indices]
        batch_y_auth = self.y_auth[indices]
        batch_y_pos = self.y_pos[indices]

        if self.augment:
            # Faster augmentation - use numpy operations directly
            augmented_X = np.array([self.datagen.random_transform(x) for x in batch_X])
        else:
            augmented_X = batch_X

        return augmented_X, {
            'ensemble_classification': batch_y_class,
            'final_authenticity': batch_y_auth,
            'positions': batch_y_pos
        }

    def on_epoch_end(self):
        self.indices = np.arange(len(self.X))
        np.random.shuffle(self.indices)


def train_enhanced_model(data_dir: str, epochs: int = 10, batch_size: int = 32):
    """Train enhanced model with position prediction"""
    print("="*60)
    print("ENHANCED TRAINING: Classification + Authenticity + Positions")
    print("="*60)
    
    # Load dataset
    print("\n[INFO] Loading dataset with position labels...")
    print("   [INFO] Using learned positions directly (fast mode - no OCR per image)")
    print("   [INFO] This will be much faster than extracting positions from each image...")
    (X_train, y_train_class, y_train_auth, y_train_pos), \
    (X_val, y_val_class, y_val_auth, y_val_pos), \
    (X_test, y_test_class, y_test_auth, y_test_pos) = load_dataset_with_positions(data_dir)
    
    print(f"\n[SUCCESS] Train: {len(X_train)} images loaded")
    print(f"[SUCCESS] Val: {len(X_val)} images loaded")
    print(f"[SUCCESS] Test: {len(X_test)} images loaded")
    
    # Create model
    print("\n[INFO] Creating enhanced ensemble model...")
    model = create_enhanced_ensemble_model(
        input_shape=(150, 150, 3),
        num_classes=4,
        predict_positions=True
    )
    model = compile_enhanced_model(model, learning_rate=0.001, predict_positions=True)
    
    print("\n[INFO] Model Summary:")
    model.summary()
    
    # Create data generators
    train_gen = EnhancedDataGenerator(
        X_train, y_train_class, y_train_auth, y_train_pos,
        batch_size=batch_size, augment=True
    )
    val_gen = EnhancedDataGenerator(
        X_val, y_val_class, y_val_auth, y_val_pos,
        batch_size=batch_size, augment=False
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=False,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    print("\n[INFO] Starting training...")
    import sys
    sys.stdout.flush()
    print(f"   Training on {len(X_train)} samples, {len(train_gen)} batches per epoch")
    sys.stdout.flush()
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    sys.stdout.flush()
    
    # Save model
    model_save_path = 'models/kyc_validator_enhanced.h5'
    print(f"\n[INFO] Saving model to {model_save_path}...")
    os.makedirs('models', exist_ok=True)
    try:
        model.save(model_save_path)
        print("[SUCCESS] Model saved successfully!")
    except Exception as e:
        print(f"[WARNING] Could not save full model: {e}")
        try:
            # Keras requires .weights.h5 extension for save_weights
            model.save_weights('models/kyc_validator_enhanced.weights.h5')
            print("[SUCCESS] Model weights saved!")
        except Exception as e2:
            print(f"[ERROR] Could not save weights either: {e2}")
    
    # Evaluate
    print("\n[INFO] Evaluating on test set...")
    test_gen = EnhancedDataGenerator(
        X_test, y_test_class, y_test_auth, y_test_pos,
        batch_size=batch_size, augment=False
    )
    
    test_results = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Results:")
    print(f"  Loss: {test_results[0]:.4f}")
    print(f"  Classification Accuracy: {test_results[4]:.4f}")
    print(f"  Authenticity Accuracy: {test_results[5]:.4f}")
    print(f"  Position MAE: {test_results[6]:.4f}")
    
    print("\n[SUCCESS] Enhanced training complete!")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train enhanced model with position prediction')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory with train/val/test subdirectories')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    
    args = parser.parse_args()
    
    train_enhanced_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

