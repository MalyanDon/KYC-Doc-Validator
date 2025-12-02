"""
Enhanced Ensemble CNN Model with Position Prediction
Combines classification, authenticity, AND position prediction (multi-task learning)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import VGG16
import sys
import os
sys.path.append(os.path.dirname(__file__))
from models import create_vgg16_backbone, create_custom_cnn_backbone, create_sequential_cnn_backbone


def create_enhanced_ensemble_model(input_shape=(150, 150, 3), num_classes=4, predict_positions=True):
    """
    Enhanced Ensemble Model with Position Prediction
    Multi-task learning: Classification + Authenticity + Positions
    
    Args:
        input_shape: Input image shape
        num_classes: Number of document classes
        predict_positions: If True, add position prediction head
    
    Returns:
        Model with outputs: [classification, authenticity, positions]
        positions: (photo_bbox, text_regions) - normalized coordinates
    """
    # Create individual backbones (reuse existing)
    vgg16_model = create_vgg16_backbone(input_shape, num_classes)
    custom_cnn_model = create_custom_cnn_backbone(input_shape, num_classes)
    sequential_model = create_sequential_cnn_backbone(input_shape, num_classes)
    
    # Input layer
    inputs = Input(shape=input_shape, name='input_image')
    
    # Get predictions from each backbone
    vgg16_class, vgg16_auth = vgg16_model(inputs)
    custom_class, custom_auth = custom_cnn_model(inputs)
    sequential_class, sequential_auth = sequential_model(inputs)
    
    # Average classification outputs
    class_avg = layers.Average(name='ensemble_classification')([
        vgg16_class, custom_class, sequential_class
    ])
    
    # Average authenticity outputs
    auth_avg = layers.Average(name='ensemble_authenticity')([
        vgg16_auth, custom_auth, sequential_auth
    ])
    auth_final = layers.Activation('sigmoid', name='final_authenticity')(auth_avg)
    
    # Position prediction head (NEW)
    if predict_positions:
        # Create a shared feature extractor for position prediction
        # Use VGG16 base model features (separate instance to avoid conflicts)
        vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        vgg16_base.trainable = False
        
        # Extract features from VGG16 base
        vgg_features = vgg16_base(inputs)
        pos_features = layers.GlobalAveragePooling2D()(vgg_features)
        
        # Position prediction branch
        pos_dense1 = layers.Dense(512, activation='relu', name='pos_dense1')(pos_features)
        pos_dropout1 = layers.Dropout(0.5)(pos_dense1)
        pos_dense2 = layers.Dense(256, activation='relu', name='pos_dense2')(pos_dropout1)
        pos_dropout2 = layers.Dropout(0.3)(pos_dense2)
        
        # Predict positions:
        # - Photo bbox: 4 values (x_min, y_min, x_max, y_max)
        # - Text regions: variable number, we'll predict key regions
        # For now, predict: photo (4) + name (4) + dob (4) + number (4) = 16 values
        num_position_values = 16  # 4 regions * 4 coordinates each
        positions = layers.Dense(num_position_values, activation='sigmoid', name='positions')(pos_dropout2)
        
        # Create ensemble model with positions
        ensemble = Model(
            inputs=inputs,
            outputs=[class_avg, auth_final, positions],
            name='enhanced_ensemble_kyc_validator'
        )
    else:
        # Original model without positions
        ensemble = Model(
            inputs=inputs,
            outputs=[class_avg, auth_final],
            name='ensemble_kyc_validator'
        )
    
    return ensemble


def create_position_prediction_model(input_shape=(150, 150, 3), num_regions=4):
    """
    Standalone position prediction model
    Can be trained separately or integrated into ensemble
    """
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False
    
    inputs = Input(shape=input_shape, name='input_image')
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Predict normalized positions (0-1 range)
    # Each region has 4 coordinates: (x_min, y_min, x_max, y_max)
    num_outputs = num_regions * 4
    positions = layers.Dense(num_outputs, activation='sigmoid', name='positions')(x)
    
    model = Model(inputs=inputs, outputs=positions, name='position_predictor')
    return model


def compile_enhanced_model(model, learning_rate=0.001, predict_positions=True):
    """Compile enhanced model with position prediction"""
    if predict_positions and len(model.outputs) == 3:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'ensemble_classification': 'categorical_crossentropy',
                'final_authenticity': 'binary_crossentropy',
                'positions': 'mse'  # Mean squared error for positions
            },
            loss_weights={
                'ensemble_classification': 1.0,
                'final_authenticity': 0.5,
                'positions': 0.3  # Lower weight for positions
            },
            metrics={
                'ensemble_classification': ['accuracy'],
                'final_authenticity': ['accuracy'],
                'positions': ['mae']  # Mean absolute error
            }
        )
    else:
        # Original compilation
        from models import compile_model
        return compile_model(model, learning_rate)
    
    return model


if __name__ == "__main__":
    # Test enhanced model
    print("Creating enhanced ensemble model with position prediction...")
    model = create_enhanced_ensemble_model(
        input_shape=(150, 150, 3),
        num_classes=4,
        predict_positions=True
    )
    model = compile_enhanced_model(model, predict_positions=True)
    model.summary()
    
    # Test forward pass
    import numpy as np
    test_input = np.random.rand(1, 150, 150, 3)
    predictions = model.predict(test_input, verbose=0)
    print(f"\nClassification shape: {predictions[0].shape}")
    print(f"Authenticity shape: {predictions[1].shape}")
    print(f"Positions shape: {predictions[2].shape}")

