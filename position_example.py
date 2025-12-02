"""
Simple example showing how position validation works
"""
import numpy as np

# Example: Image is 800x600 pixels
image_width = 600
image_height = 800

print("="*60)
print("POSITION VALIDATION EXAMPLE")
print("="*60)

# Step 1: Expected position (normalized 0-1)
print("\n1. EXPECTED POSITION (Normalized):")
expected = (0.05, 0.15, 0.30, 0.40)  # (x_min, y_min, x_max, y_max)
print(f"   {expected}")

# Convert to pixels
px_min = int(expected[0] * image_width)   # 0.05 * 600 = 30
py_min = int(expected[1] * image_height)  # 0.15 * 800 = 120
px_max = int(expected[2] * image_width)   # 0.30 * 600 = 180
py_max = int(expected[3] * image_height)   # 0.40 * 800 = 320

print(f"\n   In pixels (for {image_width}x{image_height} image):")
print(f"   X: {px_min} to {px_max} pixels")
print(f"   Y: {py_min} to {py_max} pixels")
print(f"   Region: ({px_min}, {py_min}) to ({px_max}, {py_max})")

# Step 2: Actual detected position (in pixels)
print("\n2. ACTUAL DETECTED POSITION (Pixels):")
actual_pixels = (50, 120, 150, 180)  # (x, y, width, height)
print(f"   {actual_pixels}")

# Convert to normalized
actual_normalized = (
    actual_pixels[0] / image_width,                    # 50/600 = 0.083
    actual_pixels[1] / image_height,                   # 120/800 = 0.15
    (actual_pixels[0] + actual_pixels[2]) / image_width,  # 200/600 = 0.33
    (actual_pixels[1] + actual_pixels[3]) / image_height  # 300/800 = 0.375
)
print(f"\n   Normalized: {tuple(round(x, 3) for x in actual_normalized)}")

# Step 3: Calculate overlap
print("\n3. CALCULATING OVERLAP:")
overlap_x_min = max(actual_normalized[0], expected[0])
overlap_y_min = max(actual_normalized[1], expected[1])
overlap_x_max = min(actual_normalized[2], expected[2])
overlap_y_max = min(actual_normalized[3], expected[3])

print(f"   Overlap region: ({overlap_x_min:.3f}, {overlap_y_min:.3f}, {overlap_x_max:.3f}, {overlap_y_max:.3f})")

if overlap_x_max > overlap_x_min and overlap_y_max > overlap_y_min:
    overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
    actual_area = (actual_normalized[2] - actual_normalized[0]) * (actual_normalized[3] - actual_normalized[1])
    overlap_ratio = overlap_area / actual_area if actual_area > 0 else 0
    
    print(f"   Overlap area: {overlap_area:.3f}")
    print(f"   Actual area: {actual_area:.3f}")
    print(f"   Overlap ratio: {overlap_ratio:.2%}")
    
    # Step 4: Validation
    print("\n4. VALIDATION:")
    if overlap_ratio >= 0.5:
        print(f"   ✅ VALID (overlap >= 50%)")
    else:
        print(f"   ❌ INVALID (overlap < 50%)")
else:
    print("   ❌ NO OVERLAP - Completely wrong position!")

print("\n" + "="*60)
