# Enhanced Dashboard Guide

## 🎯 Overview

The enhanced dashboard provides a complete web interface for document validation with:
- ✅ Document classification (Aadhaar/PAN/Fake/Other)
- ✅ Authenticity detection
- ✅ **Position-based fake detection** (NEW!)
- ✅ OCR text extraction
- ✅ Multi-method analysis
- ✅ Visual position overlays
- ✅ Complete JSON reports

---

## 🚀 How to Run

### Step 1: Start the Dashboard

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run Streamlit app
streamlit run app/streamlit_app_enhanced.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📋 Usage Steps

### 1. **Load the Model**
   - Open the sidebar (left side)
   - Select "Enhanced Model (with Position Prediction)"
   - Model path should be: `models/kyc_validator_enhanced.h5`
   - Click "🔄 Load Model"
   - Wait for "✅ Model Ready" message

### 2. **Upload Document**
   - Click "📤 Upload Document"
   - Select an image (PNG, JPG, JPEG) or PDF
   - Supported formats: Aadhaar card, PAN card images

### 3. **View Results**
   The dashboard shows results in multiple tabs:

   **📊 Overview Tab:**
   - Document type classification
   - Confidence scores
   - Class probabilities

   **🔍 Fake Detection Tab:**
   - Overall authenticity score
   - Position-based detection results
   - Breakdown of all detection methods
   - List of detected issues

   **📍 Position Analysis Tab:**
   - Detailed position validation for each element
   - Photo, Name, DOB, Document Number positions
   - Overlap ratios and deviations
   - Visual indicators for valid/suspicious positions

   **📝 OCR Results Tab:**
   - Extracted Aadhaar number
   - Extracted PAN number
   - Raw OCR text

   **📄 Full Report Tab:**
   - Complete JSON report
   - Downloadable report file

---

## 🎨 Features

### Visual Position Overlay
- **Green box:** Photo region
- **Blue box:** Name text region
- **Red box:** DOB text region
- **Yellow box:** Document number region

### Status Indicators
- **✅ Green banner:** Authentic document
- **❌ Red banner:** Fake document detected
- **⚠️ Warning icons:** Issues detected

### Detection Methods
1. **Color Analysis** - Checks document colors
2. **Border Detection** - Detects tampered borders
3. **Photo Tampering** - Detects pasted/altered photos
4. **QR Code Validation** - Validates QR codes
5. **Layout Analysis** - Checks document structure
6. **Position-Based Detection** - NEW! Validates element positions

---

## 📊 Understanding Results

### Overall Status
- **Authentic:** All checks passed, confidence >70%
- **Fake:** Multiple issues detected, confidence <70%

### Position Analysis
- **Valid:** Overlap >50%, deviation <2σ
- **Suspicious:** Low overlap or high deviation

### Confidence Scores
- **>80%:** High confidence
- **50-80%:** Medium confidence
- **<50%:** Low confidence (likely fake)

---

## 🔧 Troubleshooting

### Model Not Loading
- Check if `models/kyc_validator_enhanced.h5` exists
- Make sure you've trained the enhanced model
- Check file path in sidebar

### No Results Showing
- Make sure model is loaded first
- Check image format (PNG, JPG, JPEG, PDF)
- Try a different image

### Position Analysis Not Working
- Ensure enhanced model is loaded (not standard model)
- Check if `models/learned_aadhaar_positions.json` exists
- Verify document type is Aadhaar or PAN

---

## 📁 Files

- **`app/streamlit_app_enhanced.py`** - Enhanced dashboard
- **`app/streamlit_app.py`** - Original dashboard (standard model)
- **`DASHBOARD_GUIDE.md`** - This guide

---

## 🎉 Quick Start Example

```bash
# 1. Start dashboard
streamlit run app/streamlit_app_enhanced.py

# 2. In browser:
#    - Load enhanced model from sidebar
#    - Upload an Aadhaar card image
#    - View results in tabs
#    - Download JSON report if needed
```

---

## 💡 Tips

1. **Best Results:** Use clear, well-lit images
2. **Multiple Checks:** Review all tabs for complete analysis
3. **Position Overlay:** Helps visualize detected element positions
4. **JSON Report:** Download for programmatic use
5. **Model Type:** Always use "Enhanced Model" for position detection

---

## 🎯 What Makes It Enhanced?

Compared to the standard dashboard:
- ✅ Uses enhanced model with position prediction
- ✅ Position-based fake detection
- ✅ Visual position overlays
- ✅ Detailed position analysis
- ✅ Multi-method detection breakdown
- ✅ Better UI with tabs
- ✅ Complete JSON reports

**Ready to use!** 🚀

