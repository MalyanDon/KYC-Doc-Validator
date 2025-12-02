"""
Document Boundary Detection
Detects the actual boundaries/borders of PAN/Aadhaar cards in images
This helps normalize positions relative to the document, not the whole image
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
import math


def detect_document_boundaries(image: np.ndarray, doc_type: str = 'pan') -> Dict:
    """
    Detect document boundaries using multiple methods
    
    Returns:
        {
            'boundaries': (x, y, width, height) - document bounding box,
            'corners': [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] - four corners,
            'confidence': float,
            'method': str,
            'cropped_image': np.ndarray - cropped document region
        }
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape
    
    # Method 1: Contour-based detection (most reliable)
    result = detect_by_contours(gray, doc_type)
    if result['confidence'] > 0.7:
        return result
    
    # Method 2: Edge-based detection
    result = detect_by_edges(gray, doc_type)
    if result['confidence'] > 0.6:
        return result
    
    # Method 3: Color-based detection (for cards with distinct backgrounds)
    result = detect_by_color_segmentation(image, doc_type)
    if result['confidence'] > 0.6:
        return result
    
    # Fallback: Use full image with padding
    return {
        'boundaries': (0, 0, w, h),
        'corners': [(0, 0), (w, 0), (w, h), (0, h)],
        'confidence': 0.5,
        'method': 'fallback_full_image',
        'cropped_image': image
    }


def detect_by_contours(gray: np.ndarray, doc_type: str) -> Dict:
    """
    Detect document boundaries using contour detection
    """
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return {'confidence': 0.0, 'method': 'contours_no_contours'}
    
    # Find largest contour (likely the document)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    image_area = gray.shape[0] * gray.shape[1]
    
    # Check if contour is large enough (should be at least 30% of image)
    if area / image_area < 0.3:
        return {'confidence': 0.0, 'method': 'contours_too_small'}
    
    # Approximate contour to polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Calculate confidence based on rectangularity
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    
    # Documents should be roughly rectangular
    confidence = min(1.0, extent * 1.2)  # Boost confidence for rectangular shapes
    
    # Get four corners
    if len(approx) >= 4:
        # Sort corners: top-left, top-right, bottom-right, bottom-left
        corners = approx.reshape(-1, 2)
        corners = sort_corners(corners)
    else:
        # Use bounding box corners
        corners = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    
    return {
        'boundaries': (x, y, w, h),
        'corners': corners,
        'confidence': confidence,
        'method': 'contours',
        'contour_area_ratio': area / image_area
    }


def detect_by_edges(gray: np.ndarray, doc_type: str) -> Dict:
    """
    Detect document boundaries using edge detection and Hough lines
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=100,
        minLineLength=min(gray.shape[0], gray.shape[1])//4,
        maxLineGap=20
    )
    
    if lines is None or len(lines) < 4:
        return {'confidence': 0.0, 'method': 'edges_insufficient_lines'}
    
    # Find intersection points to get corners
    # Group lines by orientation (horizontal vs vertical)
    h_lines = []
    v_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
        
        if abs(angle) < 45 or abs(angle) > 135:
            h_lines.append((x1, y1, x2, y2))
        else:
            v_lines.append((x1, y1, x2, y2))
    
    if len(h_lines) < 2 or len(v_lines) < 2:
        return {'confidence': 0.0, 'method': 'edges_insufficient_orientation'}
    
    # Find corners by intersecting lines
    # This is simplified - find bounding box from line endpoints
    all_points = []
    for line in lines:
        all_points.append((line[0][0], line[0][1]))
        all_points.append((line[0][2], line[0][3]))
    
    if len(all_points) < 4:
        return {'confidence': 0.0, 'method': 'edges_insufficient_points'}
    
    # Get bounding box
    points = np.array(all_points)
    x = int(np.min(points[:, 0]))
    y = int(np.min(points[:, 1]))
    w = int(np.max(points[:, 0]) - x)
    h = int(np.max(points[:, 1]) - y)
    
    # Check if bounding box is reasonable
    image_area = gray.shape[0] * gray.shape[1]
    box_area = w * h
    
    if box_area / image_area < 0.3:
        return {'confidence': 0.0, 'method': 'edges_box_too_small'}
    
    confidence = min(0.8, box_area / image_area * 1.5)
    
    corners = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    
    return {
        'boundaries': (x, y, w, h),
        'corners': corners,
        'confidence': confidence,
        'method': 'edges',
        'line_count': len(lines)
    }


def detect_by_color_segmentation(image: np.ndarray, doc_type: str) -> Dict:
    """
    Detect document boundaries using color segmentation
    Useful for cards with distinct backgrounds (e.g., white PAN card on dark background)
    """
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # CORRECTED COLORS (based on actual card designs):
    # PAN Card: Light blue/green background (NOT white!)
    # Aadhaar Card: White background (NOT blue!)
    # Baal Aadhaar (children): Blue background
    if doc_type.lower() == 'pan':
        # PAN: Light blue/green background (hue 100-120 for blue-green)
        # Use wider range to catch both blue and green tints
        lower = np.array([100, 30, 150])  # Light blue-green, low saturation, medium brightness
        upper = np.array([120, 100, 255])  # Light blue-green, medium saturation, bright
    else:  # Aadhaar
        # Aadhaar: White background (low saturation, high brightness)
        # Baal Aadhaar (children) has blue, but standard is white
        # Check for white first (most common)
        lower = np.array([0, 0, 200])  # Any hue, low saturation, high brightness (white)
        upper = np.array([180, 30, 255])
    
    # Create mask
    mask = cv2.inRange(hsv, lower, upper)
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return {'confidence': 0.0, 'method': 'color_no_contours'}
    
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    image_area = image.shape[0] * image.shape[1]
    
    if area / image_area < 0.3:
        return {'confidence': 0.0, 'method': 'color_too_small'}
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Approximate corners
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx) >= 4:
        corners = sort_corners(approx.reshape(-1, 2))
    else:
        corners = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    
    confidence = min(0.75, area / image_area * 1.3)
    
    return {
        'boundaries': (x, y, w, h),
        'corners': corners,
        'confidence': confidence,
        'method': 'color_segmentation',
        'mask_area_ratio': area / image_area
    }


def sort_corners(corners: np.ndarray) -> List[Tuple[int, int]]:
    """
    Sort corners in order: top-left, top-right, bottom-right, bottom-left
    """
    # Find center
    center = np.mean(corners, axis=0)
    
    # Sort by angle from center
    def angle_from_center(point):
        return math.atan2(point[1] - center[1], point[0] - center[0])
    
    sorted_corners = sorted(corners, key=angle_from_center)
    
    # Identify corners by position
    # Top-left: smallest x+y
    # Bottom-right: largest x+y
    # Top-right: largest x-y
    # Bottom-left: smallest x-y
    
    corners_list = [(int(c[0]), int(c[1])) for c in sorted_corners]
    
    # Simple approach: sort by y first, then by x
    corners_list.sort(key=lambda c: (c[1], c[0]))
    
    # Reorder: top-left, top-right, bottom-right, bottom-left
    if len(corners_list) >= 4:
        top = sorted(corners_list[:2], key=lambda c: c[0])
        bottom = sorted(corners_list[2:], key=lambda c: c[0])
        return [top[0], top[1], bottom[1], bottom[0]]
    
    return corners_list


def crop_to_boundaries(image: np.ndarray, boundaries: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop image to document boundaries
    """
    x, y, w, h = boundaries
    h_img, w_img = image.shape[:2]
    
    # Ensure boundaries are within image
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    
    if w <= 0 or h <= 0:
        return image
    
    return image[y:y+h, x:x+w]


def normalize_position_to_boundaries(
    position: Tuple[float, float, float, float],
    boundaries: Tuple[int, int, int, int],
    original_image_size: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """
    Normalize position coordinates relative to document boundaries, not whole image
    
    Args:
        position: Normalized position (0-1) relative to original image
        boundaries: Document boundaries (x, y, width, height) in original image
        original_image_size: (width, height) of original image
    
    Returns:
        Normalized position (0-1) relative to document boundaries
    """
    x_norm, y_norm, w_norm, h_norm = position
    x_bound, y_bound, w_bound, h_bound = boundaries
    orig_w, orig_h = original_image_size
    
    # Convert normalized position to pixel coordinates in original image
    x_pixel = x_norm * orig_w
    y_pixel = y_norm * orig_h
    w_pixel = w_norm * orig_w
    h_pixel = h_norm * orig_h
    
    # Adjust relative to document boundaries
    x_relative = x_pixel - x_bound
    y_relative = y_pixel - y_bound
    
    # Normalize relative to document size
    x_norm_new = x_relative / w_bound if w_bound > 0 else 0
    y_norm_new = y_relative / h_bound if h_bound > 0 else 0
    w_norm_new = w_pixel / w_bound if w_bound > 0 else 0
    h_norm_new = h_pixel / h_bound if h_bound > 0 else 0
    
    return (x_norm_new, y_norm_new, w_norm_new, h_norm_new)


def draw_boundaries(image: np.ndarray, boundaries: Dict) -> np.ndarray:
    """
    Draw detected boundaries on image for visualization
    """
    img_copy = image.copy()
    x, y, w, h = boundaries['boundaries']
    corners = boundaries['corners']
    
    # Draw bounding box
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Draw corners
    for i, (cx, cy) in enumerate(corners):
        cv2.circle(img_copy, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(img_copy, str(i), (cx+5, cy-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Draw method and confidence
    cv2.putText(img_copy, f"{boundaries['method']} ({boundaries['confidence']:.2f})",
               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return img_copy

