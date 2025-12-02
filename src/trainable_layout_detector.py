"""
Trainable Layout Detector - Learns positions from real documents
Uses object detection/keypoint detection to learn where elements should be
"""

import cv2
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import MobileNetV2
import pytesseract


class LayoutPositionLearner:
    """
    Learns document layout positions from training data
    Can be trained on real documents to learn where elements should be
    """
    
    def __init__(self, doc_type: str = 'aadhaar'):
        self.doc_type = doc_type
        self.learned_positions = None
        self.model = None
        self.position_stats = {}
    
    def learn_from_annotations(self, annotations_dir: str):
        """
        Learn positions from annotated training data
        Expected format: JSON files with bounding box annotations
        """
        annotations = []
        
        # Load all annotation files
        for json_file in Path(annotations_dir).glob('*.json'):
            with open(json_file, 'r') as f:
                data = json.load(f)
                annotations.append(data)
        
        if not annotations:
            print("⚠️  No annotations found. Using default positions.")
            return self._get_default_positions()
        
        # Calculate average positions from annotations
        photo_positions = []
        text_positions = {}
        
        for ann in annotations:
            image_size = ann.get('image_size', (600, 800))  # (width, height)
            w, h = image_size
            
            # Collect photo positions
            if 'photo' in ann:
                photo = ann['photo']
                # Normalize
                x_min = photo['x'] / w
                y_min = photo['y'] / h
                x_max = (photo['x'] + photo['width']) / w
                y_max = (photo['y'] + photo['height']) / h
                photo_positions.append((x_min, y_min, x_max, y_max))
            
            # Collect text positions
            if 'text_regions' in ann:
                for label, region in ann['text_regions'].items():
                    if label not in text_positions:
                        text_positions[label] = []
                    
                    x_min = region['x'] / w
                    y_min = region['y'] / h
                    x_max = (region['x'] + region['width']) / w
                    y_max = (region['y'] + region['height']) / h
                    text_positions[label].append((x_min, y_min, x_max, y_max))
        
        # Calculate average positions
        learned_layout = {}
        
        if photo_positions:
            # Average photo position
            avg_photo = np.mean(photo_positions, axis=0)
            learned_layout['photo_region'] = tuple(avg_photo.tolist())
            self.position_stats['photo'] = {
                'mean': avg_photo.tolist(),
                'std': np.std(photo_positions, axis=0).tolist(),
                'count': len(photo_positions)
            }
        
        # Average text positions
        learned_layout['text_regions'] = {}
        for label, positions in text_positions.items():
            if positions:
                avg_pos = np.mean(positions, axis=0)
                std_pos = np.std(positions, axis=0)
                learned_layout['text_regions'][label] = tuple(avg_pos.tolist())
                self.position_stats[label] = {
                    'mean': avg_pos.tolist(),
                    'std': std_pos.tolist(),
                    'count': len(positions)
                }
        
        self.learned_positions = learned_layout
        return learned_layout
    
    def learn_from_images(self, images_dir: str, auto_annotate: bool = True):
        """
        Learn positions by automatically detecting elements in real documents
        Uses face detection + OCR to automatically annotate
        """
        images = list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.png'))
        
        if not images:
            print("⚠️  No images found. Using default positions.")
            return self._get_default_positions()
        
        photo_positions = []
        text_positions = {}
        
        print(f"📚 Learning from {len(images)} images...")
        
        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            h, w = image.shape[:2]
            
            # Auto-detect photo
            photo_region = self._detect_photo_auto(image)
            if photo_region:
                x, y, pw, ph = photo_region
                # Normalize
                x_min = x / w
                y_min = y / h
                x_max = (x + pw) / w
                y_max = (y + ph) / h
                photo_positions.append((x_min, y_min, x_max, y_max))
            
            # Auto-detect text regions
            if auto_annotate:
                texts = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                detected_texts = self._group_text_by_regions(texts, w, h)
                
                for label, region in detected_texts.items():
                    if label not in text_positions:
                        text_positions[label] = []
                    
                    x_min = region['x'] / w
                    y_min = region['y'] / h
                    x_max = (region['x'] + region['width']) / w
                    y_max = (region['y'] + region['height']) / h
                    text_positions[label].append((x_min, y_min, x_max, y_max))
        
        # Calculate averages
        learned_layout = {}
        
        if photo_positions:
            avg_photo = np.mean(photo_positions, axis=0)
            learned_layout['photo_region'] = tuple(avg_photo.tolist())
            self.position_stats['photo'] = {
                'mean': avg_photo.tolist(),
                'std': np.std(photo_positions, axis=0).tolist(),
                'count': len(photo_positions)
            }
            print(f"✅ Learned photo position from {len(photo_positions)} samples")
        
        for label, positions in text_positions.items():
            if positions:
                avg_pos = np.mean(positions, axis=0)
                learned_layout['text_regions'] = learned_layout.get('text_regions', {})
                learned_layout['text_regions'][label] = tuple(avg_pos.tolist())
                self.position_stats[label] = {
                    'mean': avg_pos.tolist(),
                    'std': np.std(positions, axis=0).tolist(),
                    'count': len(positions)
                }
                print(f"✅ Learned {label} position from {len(positions)} samples")
        
        self.learned_positions = learned_layout
        return learned_layout
    
    def _detect_photo_auto(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Auto-detect photo using face detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, fw, fh = largest_face
                
                # Expand to full photo
                margin = 20
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(image.shape[1] - x, fw + 2 * margin)
                h = min(image.shape[0] - y, fh + 2 * margin)
                
                return (x, y, w, h)
        except:
            pass
        
        return None
    
    def _group_text_by_regions(self, ocr_data: Dict, image_width: int, image_height: int) -> Dict:
        """Group OCR text into logical regions (name, dob, etc.)"""
        # This is a simplified version - can be enhanced with ML
        regions = {}
        
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            if not text or int(ocr_data['conf'][i]) < 30:
                continue
            
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]
            
            # Simple heuristics to label regions
            # Can be improved with ML-based classification
            normalized_y = y / image_height
            
            if normalized_y < 0.35:
                if 'name' not in regions:
                    regions['name'] = {'x': x, 'y': y, 'width': w, 'height': h}
                else:
                    # Expand region
                    regions['name']['width'] = max(regions['name']['width'], x + w - regions['name']['x'])
            elif 0.35 <= normalized_y < 0.50:
                if 'dob' not in regions:
                    regions['dob'] = {'x': x, 'y': y, 'width': w, 'height': h}
            elif 0.50 <= normalized_y < 0.65:
                # Check if it's a number (Aadhaar/PAN)
                if any(char.isdigit() for char in text):
                    if 'aadhaar_number' not in regions and 'pan_number' not in regions:
                        regions['document_number'] = {'x': x, 'y': y, 'width': w, 'height': h}
        
        return regions
    
    def _get_default_positions(self):
        """Fallback to default positions if no training data"""
        if self.doc_type.lower() == 'aadhaar':
            return {
                'photo_region': (0.05, 0.15, 0.30, 0.40),
                'text_regions': {
                    'name': (0.35, 0.20, 0.95, 0.30),
                    'dob': (0.35, 0.30, 0.70, 0.40),
                    'aadhaar_number': (0.35, 0.50, 0.95, 0.60),
                }
            }
        else:  # PAN
            return {
                'photo_region': (0.70, 0.15, 0.95, 0.40),
                'text_regions': {
                    'name': (0.05, 0.20, 0.65, 0.30),
                    'pan_number': (0.05, 0.50, 0.60, 0.60),
                }
            }
    
    def save_learned_positions(self, filepath: str):
        """Save learned positions to file"""
        data = {
            'doc_type': self.doc_type,
            'learned_positions': self.learned_positions,
            'position_stats': self.position_stats
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved learned positions to {filepath}")
    
    def load_learned_positions(self, filepath: str):
        """Load learned positions from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.doc_type = data.get('doc_type', self.doc_type)
        self.learned_positions = data.get('learned_positions')
        self.position_stats = data.get('position_stats', {})
        print(f"✅ Loaded learned positions from {filepath}")
        return self.learned_positions
    
    def get_positions(self):
        """Get learned positions (or defaults if not trained)"""
        if self.learned_positions:
            return self.learned_positions
        return self._get_default_positions()


def create_position_detection_model(input_shape=(224, 224, 3), num_keypoints=8):
    """
    Create a CNN model to detect keypoints/positions in documents
    Can be trained to predict where elements should be
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Predict keypoints (normalized coordinates)
    # For photo: 4 values (x_min, y_min, x_max, y_max)
    # For text regions: multiple sets of 4 values
    keypoints = layers.Dense(num_keypoints * 2, activation='sigmoid', name='keypoints')(x)
    
    model = Model(inputs=inputs, outputs=keypoints, name='position_detector')
    return model


def train_position_detector(images_dir: str, annotations_dir: str, 
                           output_model_path: str = 'models/position_detector.h5',
                           epochs: int = 20):
    """
    Train a model to detect element positions in documents
    """
    # This would require annotated training data
    # For now, we use the learning-from-images approach
    print("📚 Training position detector...")
    print("💡 Note: Full training requires annotated bounding boxes")
    print("💡 Using auto-detection approach instead...")
    
    learner = LayoutPositionLearner()
    learned = learner.learn_from_images(images_dir, auto_annotate=True)
    learner.save_learned_positions('models/learned_positions.json')
    
    return learned


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("TRAINABLE LAYOUT POSITION LEARNER")
    print("="*60)
    
    # Method 1: Learn from images (auto-annotation)
    learner = LayoutPositionLearner(doc_type='aadhaar')
    
    # If you have real document images
    # learned = learner.learn_from_images('data/train/aadhaar/')
    # learner.save_learned_positions('models/learned_aadhaar_positions.json')
    
    # Method 2: Load pre-learned positions
    # learner.load_learned_positions('models/learned_aadhaar_positions.json')
    
    # Get positions
    positions = learner.get_positions()
    print(f"\n📐 Learned Positions:")
    print(f"Photo: {positions.get('photo_region')}")
    print(f"Text Regions: {positions.get('text_regions')}")

