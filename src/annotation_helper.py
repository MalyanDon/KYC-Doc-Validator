"""
Annotation Helper - Tool to create training data for position learning
Helps annotate real documents to train the position detector
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class DocumentAnnotator:
    """
    Interactive tool to annotate document elements
    Creates training data for position learning
    """
    
    def __init__(self):
        self.annotations = []
        self.current_image = None
        self.current_path = None
        self.drawing = False
        self.start_point = None
        self.current_box = None
        self.boxes = {}  # label -> [(x, y, w, h), ...]
        self.current_label = 'photo'
    
    def annotate_image(self, image_path: str, output_path: Optional[str] = None):
        """
        Interactive annotation tool
        Click and drag to draw bounding boxes
        """
        self.current_path = image_path
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load {image_path}")
            return None
        
        self.current_image = image.copy()
        h, w = image.shape[:2]
        
        print("\n" + "="*60)
        print("DOCUMENT ANNOTATION TOOL")
        print("="*60)
        print(f"Image: {image_path}")
        print(f"Size: {w} x {h} pixels")
        print("\nInstructions:")
        print("1. Press 'p' to annotate PHOTO")
        print("2. Press 'n' to annotate NAME")
        print("3. Press 'd' to annotate DOB")
        print("4. Press 'a' to annotate AADHAAR NUMBER")
        print("5. Click and drag to draw bounding box")
        print("6. Press 's' to save annotation")
        print("7. Press 'q' to quit")
        print("="*60)
        
        # Create window
        cv2.namedWindow('Annotation Tool')
        cv2.setMouseCallback('Annotation Tool', self._mouse_callback)
        
        while True:
            display_image = self._draw_boxes(self.current_image.copy())
            cv2.imshow('Annotation Tool', display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_annotation(image_path, w, h, output_path)
                break
            elif key == ord('p'):
                self.current_label = 'photo'
                print("📸 Now annotating: PHOTO")
            elif key == ord('n'):
                self.current_label = 'name'
                print("📝 Now annotating: NAME")
            elif key == ord('d'):
                self.current_label = 'dob'
                print("📅 Now annotating: DOB")
            elif key == ord('a'):
                self.current_label = 'aadhaar_number'
                print("🆔 Now annotating: AADHAAR NUMBER")
        
        cv2.destroyAllWindows()
        return self.boxes
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_box = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.start_point:
                self.current_box = (self.start_point[0], self.start_point[1], 
                                  x - self.start_point[0], y - self.start_point[1])
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # Normalize coordinates
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1, x2)
                y_max = max(y1, y2)
                
                width = x_max - x_min
                height = y_max - y_min
                
                if width > 10 and height > 10:  # Minimum size
                    if self.current_label not in self.boxes:
                        self.boxes[self.current_label] = []
                    
                    self.boxes[self.current_label].append({
                        'x': x_min,
                        'y': y_min,
                        'width': width,
                        'height': height
                    })
                    print(f"✅ Added {self.current_label} box: ({x_min}, {y_min}, {width}, {height})")
                
                self.drawing = False
                self.current_box = None
    
    def _draw_boxes(self, image):
        """Draw all annotated boxes on image"""
        colors = {
            'photo': (0, 255, 0),      # Green
            'name': (255, 0, 0),       # Blue
            'dob': (0, 0, 255),        # Red
            'aadhaar_number': (255, 255, 0),  # Cyan
        }
        
        # Draw saved boxes
        for label, boxes in self.boxes.items():
            color = colors.get(label, (255, 255, 255))
            for box in boxes:
                x, y, w, h = box['x'], box['y'], box['width'], box['height']
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, label, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw current box being drawn
        if self.current_box:
            x, y, w, h = self.current_box
            color = colors.get(self.current_label, (255, 255, 255))
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        return image
    
    def _save_annotation(self, image_path: str, image_width: int, image_height: int, 
                        output_path: Optional[str] = None):
        """Save annotation to JSON file"""
        annotation = {
            'image_path': str(image_path),
            'image_size': [image_width, image_height],
            'photo': self.boxes.get('photo', [{}])[0] if 'photo' in self.boxes else None,
            'text_regions': {}
        }
        
        for label in ['name', 'dob', 'aadhaar_number', 'pan_number']:
            if label in self.boxes and self.boxes[label]:
                annotation['text_regions'][label] = self.boxes[label][0]
        
        if output_path is None:
            output_path = str(Path(image_path).with_suffix('.json'))
        
        with open(output_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        print(f"\n✅ Saved annotation to {output_path}")
        return annotation


def batch_annotate(images_dir: str, output_dir: Optional[str] = None):
    """
    Batch annotation tool - annotate multiple images
    """
    images = list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.png'))
    
    if not images:
        print(f"❌ No images found in {images_dir}")
        return
    
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    annotator = DocumentAnnotator()
    
    print(f"\n📚 Found {len(images)} images to annotate")
    print("Annotate each image, then press 's' to save and move to next")
    
    for i, img_path in enumerate(images, 1):
        print(f"\n{'='*60}")
        print(f"Image {i}/{len(images)}: {img_path.name}")
        print(f"{'='*60}")
        
        output_path = None
        if output_dir:
            output_path = Path(output_dir) / f"{img_path.stem}.json"
        
        annotator.boxes = {}  # Reset for each image
        annotator.annotate_image(str(img_path), str(output_path) if output_path else None)
        
        response = input("\nContinue to next image? (y/n): ")
        if response.lower() != 'y':
            break


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image: python annotation_helper.py <image_path>")
        print("  Batch: python annotation_helper.py --batch <images_dir> [output_dir]")
        sys.exit(1)
    
    if sys.argv[1] == '--batch':
        images_dir = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        batch_annotate(images_dir, output_dir)
    else:
        image_path = sys.argv[1]
        annotator = DocumentAnnotator()
        annotator.annotate_image(image_path)

