"""
Ensemble CNN Model for KYC Document Classification
Combines VGG16, Custom CNN, and Lightweight Sequential CNN
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import VGG16


def create_vgg16_backbone(input_shape=(150, 150, 3), num_classes=4, use_flatten=False, fine_tune_last_4=False):
    """
    Backbone 1: VGG16-based classifier
    Adapted from Magnum-Opus repository with improvements
    
    Args:
        use_flatten: If True, use Flatten (like Magnum-Opus), else GlobalAveragePooling
        fine_tune_last_4: If True, unfreeze last 4 layers for fine-tuning (like Magnum-Opus)
    """
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Fine-tuning option: unfreeze last 4 layers (Magnum-Opus approach)
    if fine_tune_last_4:
        for layer in base_model.layers[:-4]:
            layer.trainable = False
        for layer in base_model.layers[-4:]:
            layer.trainable = True
    else:
        base_model.trainable = False  # Freeze base initially
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    
    # Option to use Flatten (sometimes better for documents) or GlobalAveragePooling
    if use_flatten:
        x = layers.Flatten()(x)
        # Magnum-Opus uses Dense(1024) after Flatten
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
    else:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
    
    # Classification head
    class_output = layers.Dense(num_classes, activation='softmax', name='classification')(x)
    
    # Authenticity head (binary)
    auth_output = layers.Dense(1, activation='sigmoid', name='authenticity')(x)
    
    model = Model(inputs=inputs, outputs=[class_output, auth_output], name='vgg16_backbone')
    return model


def create_custom_cnn_backbone(input_shape=(150, 150, 3), num_classes=4):
    """
    Backbone 2: Custom CNN
    Adapted from documentClassification repository
    """
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 4
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Classification head
    class_output = layers.Dense(num_classes, activation='softmax', name='classification')(x)
    
    # Authenticity head
    auth_output = layers.Dense(1, activation='sigmoid', name='authenticity')(x)
    
    model = Model(inputs=inputs, outputs=[class_output, auth_output], name='custom_cnn_backbone')
    return model


def create_sequential_cnn_backbone(input_shape=(150, 150, 3), num_classes=4):
    """
    Backbone 3: Lightweight Sequential CNN
    Based on paper: Sequential model with 5 conv layers
    """
    inputs = Input(shape=input_shape)
    
    # Conv Layer 1: 32 filters
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Conv Layer 2: 64 filters
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Conv Layer 3: 128 filters
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Conv Layer 4: 256 filters
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Conv Layer 5: 512 filters
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Classification head
    class_output = layers.Dense(num_classes, activation='softmax', name='classification')(x)
    
    # Authenticity head
    auth_output = layers.Dense(1, activation='sigmoid', name='authenticity')(x)
    
    model = Model(inputs=inputs, outputs=[class_output, auth_output], name='sequential_cnn_backbone')
    return model


def create_ensemble_model(input_shape=(150, 150, 3), num_classes=4):
    """
    Ensemble Model: Fuses outputs from 3 backbones using Keras Functional API
    - Average outputs for classification
    - Sigmoid for authenticity score
    """
    # Create individual backbones
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
    
    # Apply sigmoid to authenticity (already applied in individual models, but ensure)
    auth_final = layers.Activation('sigmoid', name='final_authenticity')(auth_avg)
    
    # Create ensemble model
    ensemble = Model(
        inputs=inputs,
        outputs=[class_avg, auth_final],
        name='ensemble_kyc_validator'
    )
    
    return ensemble


def compile_model(model, learning_rate=0.001):
    """Compile the ensemble model with appropriate losses and metrics"""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'ensemble_classification': 'categorical_crossentropy',
            'final_authenticity': 'binary_crossentropy'
        },
        loss_weights={
            'ensemble_classification': 1.0,
            'final_authenticity': 0.5
        },
        metrics={
            'ensemble_classification': ['accuracy'],  # Removed top_k (needs k=5, we have 4 classes)
            'final_authenticity': ['accuracy']
        }
    )
    return model


if __name__ == "__main__":
    # Test model creation
    print("Creating ensemble model...")
    model = create_ensemble_model(input_shape=(150, 150, 3), num_classes=4)
    model = compile_model(model)
    model.summary()
    
    # Test forward pass
    import numpy as np
    test_input = np.random.rand(1, 150, 150, 3)
    predictions = model.predict(test_input, verbose=0)
    print(f"\nClassification shape: {predictions[0].shape}")
    print(f"Authenticity shape: {predictions[1].shape}")

