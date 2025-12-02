"""
Training Script for KYC Document Validator
Handles dataset loading, training, and evaluation
"""

import os
# macOS TensorFlow workaround - set before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Multi-core optimization - use all available cores
# Get number of CPU cores
import multiprocessing
num_cores = multiprocessing.cpu_count()
os.environ['TF_NUM_INTEROP_THREADS'] = str(num_cores)  # Parallel operations
os.environ['TF_NUM_INTRAOP_THREADS'] = str(num_cores)  # Within-operation parallelism
os.environ['OMP_NUM_THREADS'] = str(num_cores)  # OpenMP threads

import numpy as np
import cv2

# Import TensorFlow after setting environment
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Configure TensorFlow to use all cores
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    tf.config.threading.set_intra_op_parallelism_threads(num_cores)
    
    print(f"✅ TensorFlow configured for {num_cores} CPU cores")
    print(f"   Inter-op threads: {tf.config.threading.get_inter_op_parallelism_threads()}")
    print(f"   Intra-op threads: {tf.config.threading.get_intra_op_parallelism_threads()}")
except Exception as e:
    print(f"⚠️  TensorFlow import issue: {e}")
    print("💡 This is a known macOS issue. Training may still work.")
    import keras
    from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from models import create_ensemble_model, compile_model
import albumentations as A
# Note: ToTensorV2 is for PyTorch, we don't need it for TensorFlow/Keras


def load_and_preprocess_image(image_path: str, target_size: tuple = (150, 150)) -> np.ndarray:
    """
    Load and preprocess image for training
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    
    return image


def create_augmentation_pipeline():
    """
    Create augmentation pipeline using Albumentations
    For generating synthetic fake documents
    """
    return A.Compose([
        A.RandomRotate90(p=0.3),
        A.Flip(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.GaussNoise(p=0.2),
        A.ElasticTransform(p=0.2, alpha=50, sigma=5),
        A.GridDistortion(p=0.2),
        A.OpticalDistortion(p=0.2),
        A.ShiftScaleRotate(p=0.3),
        A.CoarseDropout(p=0.2, max_holes=8, max_height=16, max_width=16),
    ])


def load_dataset(data_dir: str, target_size: tuple = (150, 150)) -> tuple:
    """
    Load dataset from directory structure
    Expected structure: data_dir/{train,val,test}/{aadhaar,pan,fake,other}/
    """
    X_train, y_train_class, y_train_auth = [], [], []
    X_val, y_val_class, y_val_auth = [], [], []
    X_test, y_test_class, y_test_auth = [], [], []
    
    class_mapping = {'aadhaar': 0, 'pan': 1, 'fake': 2, 'other': 3}
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist, skipping...")
            continue
        
        for class_name, class_idx in class_mapping.items():
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist, skipping...")
                continue
            
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                try:
                    image = load_and_preprocess_image(img_path, target_size)
                    
                    # One-hot encode class
                    class_onehot = keras.utils.to_categorical(class_idx, num_classes=4)
                    
                    # Authenticity: 1 for real (aadhaar, pan, other), 0 for fake
                    auth_label = 1.0 if class_name != 'fake' else 0.0
                    
                    if split == 'train':
                        X_train.append(image)
                        y_train_class.append(class_onehot)
                        y_train_auth.append(auth_label)
                    elif split == 'val':
                        X_val.append(image)
                        y_val_class.append(class_onehot)
                        y_val_auth.append(auth_label)
                    else:  # test
                        X_test.append(image)
                        y_test_class.append(class_onehot)
                        y_test_auth.append(auth_label)
                
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
    
    # Debug: Check what we have
    print(f"\n🔍 Debug - Before conversion:")
    print(f"   X_train length: {len(X_train)}")
    print(f"   y_train_class length: {len(y_train_class)}")
    print(f"   y_train_auth length: {len(y_train_auth)}")
    
    # Convert to numpy arrays
    def convert_to_array(X, y_class, y_auth):
        if len(X) == 0:
            print(f"   ⚠️  Empty X array")
            return None, None, None
        if len(y_class) == 0:
            print(f"   ⚠️  Empty y_class array (X has {len(X)} items)")
            return None, None, None
        if len(y_auth) == 0:
            print(f"   ⚠️  Empty y_auth array (X has {len(X)} items)")
            return None, None, None
        
        X_arr = np.array(X)
        y_class_arr = np.array(y_class)
        y_auth_arr = np.array(y_auth)
        
        print(f"   ✅ Converted: X={X_arr.shape}, y_class={y_class_arr.shape}, y_auth={y_auth_arr.shape}")
        
        return X_arr, y_class_arr, y_auth_arr
    
    train_data = convert_to_array(X_train, y_train_class, y_train_auth)
    val_data = convert_to_array(X_val, y_val_class, y_val_auth)
    test_data = convert_to_array(X_test, y_test_class, y_test_auth)
    
    # Check if we have data
    if train_data[0] is None or len(train_data[0]) == 0:
        raise ValueError("No training data found! Please add images to data/train/")
    
    print(f"\n✅ Dataset loaded successfully:")
    print(f"   Train: {len(train_data[0])} images, labels: {train_data[1].shape}")
    if val_data[0] is not None:
        print(f"   Val: {len(val_data[0])} images, labels: {val_data[1].shape}")
    if test_data[0] is not None:
        print(f"   Test: {len(test_data[0])} images, labels: {test_data[1].shape}")
    
    return train_data, val_data, test_data


def create_data_generator(X, y_class, y_auth, batch_size=32, augment=False):
    """
    Create custom data generator for multi-output model
    ImageDataGenerator doesn't support multi-output, so we create a custom generator
    """
    # Ensure arrays are numpy arrays and have correct shape
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y_class, np.ndarray):
        y_class = np.array(y_class)
    if not isinstance(y_auth, np.ndarray):
        y_auth = np.array(y_auth)
    
    # Check shapes match
    if len(X) != len(y_class) or len(X) != len(y_auth):
        raise ValueError(f"Shape mismatch: X={len(X)}, y_class={len(y_class)}, y_auth={len(y_auth)}")
    
    # Reshape y_auth if needed (should be 1D or 2D with shape (n, 1))
    if y_auth.ndim == 1:
        y_auth = y_auth.reshape(-1, 1)
    
    # Create indices for shuffling
    indices = np.arange(len(X))
    
    def generator():
        while True:
            # Shuffle indices each epoch
            np.random.shuffle(indices)
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = X[batch_indices]
                batch_y_class = y_class[batch_indices]
                batch_y_auth = y_auth[batch_indices]
                
                # Apply augmentation if needed
                if augment:
                    # Simple augmentation using ImageDataGenerator on images only
                    aug_datagen = ImageDataGenerator(
                        rotation_range=20,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='nearest'
                    )
                    # Apply augmentation to batch
                    aug_iter = aug_datagen.flow(batch_X, shuffle=False, batch_size=len(batch_X))
                    batch_X = next(aug_iter)
                
                yield batch_X, {'ensemble_classification': batch_y_class, 'final_authenticity': batch_y_auth}
    
    return generator()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """
    Plot and save confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def train_model(data_dir: str, epochs: int = 10, batch_size: int = 32, 
                learning_rate: float = 0.001, model_save_path: str = 'models/kyc_validator.h5'):
    """
    Main training function
    """
    print("Loading dataset...")
    train_data, val_data, test_data = load_dataset(data_dir)
    
    if train_data[0] is None:
        raise ValueError("No training data found!")
    
    X_train, y_train_class, y_train_auth = train_data
    print(f"Training samples: {len(X_train)}")
    
    if val_data[0] is not None:
        X_val, y_val_class, y_val_auth = val_data
        print(f"Validation samples: {len(X_val)}")
    else:
        X_val, y_val_class, y_val_auth = None, None, None
        print("No validation data found, using training data for validation")
    
    if test_data[0] is not None:
        X_test, y_test_class, y_test_auth = test_data
        print(f"Test samples: {len(X_test)}")
    else:
        X_test, y_test_class, y_test_auth = None, None, None
    
    # Create model
    print("\nCreating ensemble model...")
    model = create_ensemble_model(input_shape=(150, 150, 3), num_classes=4)
    model = compile_model(model, learning_rate=learning_rate)
    
    # Create data generators
    print("\n📦 Creating data generators...")
    train_gen = create_data_generator(X_train, y_train_class, y_train_auth, 
                                     batch_size=batch_size, augment=True)
    
    # Validation data - use direct arrays for validation (no generator needed)
    val_data_dict = None
    if X_val is not None:
        val_data_dict = (X_val, {'ensemble_classification': y_val_class, 'final_authenticity': y_val_auth})
    else:
        val_data_dict = (X_train, {'ensemble_classification': y_train_class, 'final_authenticity': y_train_auth})
    
    # Callbacks - No checkpoint during training to avoid HDF5 issues on macOS
    # Model will be saved at the end of training
    import os
    callbacks = [
        # Only use callbacks that don't save files during training
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=False,  # Disabled to avoid HDF5 issues
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
    
    # Calculate steps per epoch
    steps_per_epoch = len(X_train) // batch_size
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_data_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    if X_test is not None:
        print("\nEvaluating on test set...")
        test_results = model.evaluate(
            X_test,
            {'ensemble_classification': y_test_class, 'final_authenticity': y_test_auth},
            verbose=1
        )
        print(f"Test loss: {test_results[0]}")
        print(f"Test classification accuracy: {test_results[3]}")
        print(f"Test authenticity accuracy: {test_results[4]}")
        
        # Predictions for confusion matrix
        predictions = model.predict(X_test, verbose=1)
        y_pred_class = np.argmax(predictions[0], axis=1)
        y_true_class = np.argmax(y_test_class, axis=1)
        
        class_names = ['Aadhaar', 'PAN', 'Fake', 'Other']
        plot_confusion_matrix(y_true_class, y_pred_class, class_names)
        
        # Classification report - only include classes that are present
        unique_classes = np.unique(np.concatenate([y_true_class, y_pred_class]))
        present_class_names = [class_names[i] for i in unique_classes if i < len(class_names)]
        print("\nClassification Report:")
        print(classification_report(y_true_class, y_pred_class, 
                                   labels=unique_classes,
                                   target_names=present_class_names if len(present_class_names) == len(unique_classes) else None))
    
    # Save final model after training completes (do this before evaluation to ensure it's saved)
    print(f"\n💾 Saving final model to {model_save_path}...")
    import os
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    if os.path.exists(model_save_path):
        os.remove(model_save_path)  # Remove old file first
    try:
        model.save(model_save_path)
        print("✅ Model saved successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Could not save model as HDF5: {e}")
        print("   Trying to save weights only...")
        weights_path = model_save_path.replace('.h5', '_weights.h5')
        model.save_weights(weights_path)
        print(f"✅ Model weights saved to {weights_path}")
    
    # Plot training history
    try:
        plot_training_history(history)
    except Exception as e:
        print(f"⚠️  Warning: Could not plot training history: {e}")
    
    return model, history


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Classification loss
    axes[0, 0].plot(history.history['ensemble_classification_loss'], label='Train')
    axes[0, 0].plot(history.history['val_ensemble_classification_loss'], label='Val')
    axes[0, 0].set_title('Classification Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Classification accuracy
    axes[0, 1].plot(history.history['ensemble_classification_accuracy'], label='Train')
    axes[0, 1].plot(history.history['val_ensemble_classification_accuracy'], label='Val')
    axes[0, 1].set_title('Classification Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # Authenticity loss
    axes[1, 0].plot(history.history['final_authenticity_loss'], label='Train')
    axes[1, 0].plot(history.history['val_final_authenticity_loss'], label='Val')
    axes[1, 0].set_title('Authenticity Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    
    # Authenticity accuracy
    axes[1, 1].plot(history.history['final_authenticity_accuracy'], label='Train')
    axes[1, 1].plot(history.history['val_final_authenticity_accuracy'], label='Val')
    axes[1, 1].set_title('Authenticity Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history saved to {save_path}")


if __name__ == "__main__":
    import argparse
    
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
    
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_save_path=args.model_save_path
    )

