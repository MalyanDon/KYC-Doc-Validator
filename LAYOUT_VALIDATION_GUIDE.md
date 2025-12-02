# Layout-Based Validation Guide

## 🎯 Smart Position-Based Fake Detection

Instead of just checking for white paper, we now use **intelligent position-based validation** that:

1. ✅ **Checks photo position** - Verifies photo is in correct location for document type
2. ✅ **Verifies photo authenticity** - Detects if photo is pasted/tampered
3. ✅ **Validates text positions** - Ensures text elements are in correct positions
4. ✅ **Uses OCR with positions** - Extracts data and validates layout

---

## 📐 How It Works

### 1. Photo Position Validation

**For Aadhaar Cards:**
- Photo should be in **top-left region** (5%-30% from left, 15%-40% from top)
- If photo is in wrong position → **Flagged as fake**

**For PAN Cards:**
- Photo should be in **top-right region** (70%-95% from left, 15%-40% from top)
- If photo is in wrong position → **Flagged as fake**

**Detection Method:**
```python
# Detects photo using:
1. Face detection (if available)
2. Region analysis (rectangular regions with different characteristics)
3. Position validation against expected layout
```

### 2. Photo Authenticity Verification

Checks if photo is **real** (not pasted/tampered):

1. **Sharp Edge Detection**
   - Pasted photos have sharp edges around borders
   - Checks border regions for sharp transitions

2. **Lighting Consistency**
   - Real photos match document lighting
   - Pasted photos often have different lighting

3. **Compression Artifact Analysis**
   - Different compression patterns indicate different sources
   - Uses frequency domain analysis

4. **Photo Quality Check**
   - Real photos have reasonable quality
   - Too uniform = suspicious

### 3. Text Position Validation

**Expected Text Regions for Aadhaar:**
- Name: 35%-95% from left, 20%-30% from top
- DOB: 35%-70% from left, 30%-40% from top
- Gender: 35%-60% from left, 40%-50% from top
- Aadhaar Number: 35%-95% from left, 50%-60% from top
- Address: 5%-95% from left, 60%-85% from top

**Validation:**
- Extracts text with positions using OCR
- Checks if text appears in expected regions
- Flags missing or misplaced text

### 4. OCR with Position Data

**Enhanced OCR Extraction:**
```python
extract_text_with_positions(image)
# Returns:
[
    {
        'text': 'John Doe',
        'x': 200,
        'y': 100,
        'width': 150,
        'height': 30,
        'confidence': 95
    },
    ...
]
```

---

## 🔍 Detection Flow

```
1. Detect Photo Region
   ↓
2. Validate Photo Position (correct for document type?)
   ↓
3. Verify Photo Authenticity (real or pasted?)
   ↓
4. Extract Text with Positions (OCR)
   ↓
5. Validate Text Positions (in correct regions?)
   ↓
6. Combine Results → Authenticity Score
```

---

## 🚨 Issues Detected

### Photo Issues:
- `no_photo_detected` - No photo found
- `photo_wrong_position` - Photo not in expected location
- `sharp_edges_around_photo` - Indicates pasting
- `inconsistent_lighting` - Photo lighting doesn't match document
- `compression_artifact_mismatch` - Different compression patterns
- `low_photo_quality` - Photo quality too low

### Text Issues:
- `missing_text_name` - Name not found in expected region
- `missing_text_dob` - DOB not found in expected region
- `missing_text_aadhaar_number` - Aadhaar number not in expected region
- `text_position_mismatch` - Text in wrong positions

### Layout Issues:
- `unknown_document_type` - Can't validate layout
- `layout_validation_failed` - Overall layout invalid

---

## 💻 Usage

### Basic Usage:
```python
from src.layout_validator import comprehensive_layout_validation
import cv2

image = cv2.imread('document.jpg')
result = comprehensive_layout_validation(image, 'aadhaar', ocr_text)

print(f"Valid: {result['is_valid']}")
print(f"Confidence: {result['confidence']}")
print(f"Photo Position Valid: {result['photo_position_valid']}")
print(f"Photo Authentic: {result['photo_authentic']}")
print(f"Text Positions Valid: {result['text_positions_valid']}")
print(f"Issues: {result['issues']}")
```

### Integrated with Fake Detection:
```python
from src.fake_detector import comprehensive_fake_detection

# Layout validation is automatically included
result = comprehensive_fake_detection(
    image, 
    doc_type='aadhaar',
    ocr_text='...',
    use_layout_validation=True  # Enable layout validation
)

# Check layout results
layout_result = result['detailed_results']['layout_validation']
```

---

## 📊 Example Results

### Real Document:
```json
{
  "is_valid": true,
  "confidence": 0.92,
  "photo_detected": true,
  "photo_position_valid": true,
  "photo_authentic": true,
  "text_positions_valid": true,
  "issues": []
}
```

### Fake Document (Wrong Photo Position):
```json
{
  "is_valid": false,
  "confidence": 0.45,
  "photo_detected": true,
  "photo_position_valid": false,
  "photo_authentic": false,
  "text_positions_valid": false,
  "issues": [
    "photo_wrong_position",
    "sharp_edges_around_photo",
    "missing_text_name",
    "text_position_mismatch"
  ]
}
```

---

## 🎓 Key Advantages

1. ✅ **Works on any background** - Not limited to white paper
2. ✅ **Position-based validation** - Checks actual document structure
3. ✅ **Photo authenticity** - Detects pasted/tampered photos
4. ✅ **Text position validation** - Ensures proper layout
5. ✅ **OCR integration** - Uses extracted data for validation
6. ✅ **Document-type specific** - Different layouts for Aadhaar vs PAN

---

## 🔧 Configuration

### Adjust Layout Definitions:
Edit `layout_validator.py` to customize expected positions:

```python
AADHAAR_LAYOUT = DocumentLayout(
    doc_type='aadhaar',
    photo_region=(0.05, 0.15, 0.30, 0.40),  # Adjust as needed
    text_regions=[...],  # Customize text regions
    security_features=[...]
)
```

### Sensitivity Tuning:
- Photo position overlap threshold (default: 50%)
- Lighting difference threshold (default: 40)
- Compression artifact threshold (default: 2.0)

---

## 📝 Next Steps

1. **Collect real document samples** to refine layout definitions
2. **Test on various fake documents** to improve detection
3. **Fine-tune position thresholds** based on results
4. **Add more document types** (Voter ID, Driving License, etc.)

---

**This approach is much smarter and works regardless of background!** 🎉

