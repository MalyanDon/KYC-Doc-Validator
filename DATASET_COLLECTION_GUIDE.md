# 📸 Dataset Collection Guide

## Current Status: ❌ NO IMAGES FOUND

All data directories are **empty**. You need to collect and add images before training.

---

## 📋 What You Need

### Minimum Dataset (For Testing):
- **Aadhaar:** 20-30 images (train) + 5-10 (val) + 5-10 (test)
- **PAN:** 20-30 images (train) + 5-10 (val) + 5-10 (test)
- **Fake:** 10-20 images (train) + 3-5 (val) + 3-5 (test)
- **Other:** 10-20 images (train) + 3-5 (val) + 3-5 (test)
- **Total:** ~100-150 images minimum

### Recommended Dataset (For Good Results):
- **Aadhaar:** 200+ images per split (train/val/test)
- **PAN:** 200+ images per split
- **Fake:** 100+ images per split
- **Other:** 50+ images per split
- **Total:** ~1,500+ images

---

## 🎯 Where to Get Images

### Option 1: Use Your Own Documents (Best Quality)
**Aadhaar & PAN Cards:**
- Scan or photograph real documents (with permission)
- Ensure good lighting and clarity
- Remove sensitive information if needed (blur numbers)
- Save as JPG or PNG format

**Steps:**
1. Scan/photograph documents
2. Save to a temporary folder
3. Copy to appropriate `data/` folders

### Option 2: Public Datasets (If Available)
Search for:
- Indian ID document datasets
- Aadhaar card datasets
- PAN card datasets
- Document verification datasets

**Note:** Ensure compliance with data privacy regulations.

### Option 3: Create Synthetic Fake Documents
Use data augmentation to create fake documents:

```python
# Example: Create synthetic fakes from real documents
import cv2
import numpy as np
import albumentations as A

# Load a real document
img = cv2.imread('real_aadhaar.jpg')

# Apply transformations to create "fake" versions
transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.MotionBlur(blur_limit=3, p=0.3),
])

fake_img = transform(image=img)['image']
cv2.imwrite('data/train/fake/synthetic_fake_1.jpg', fake_img)
```

### Option 4: Web Scraping (Use with Caution)
- Only if legally permitted
- Respect robots.txt and terms of service
- Ensure data privacy compliance
- Remove sensitive information

---

## 📁 How to Organize Images

### Directory Structure:
```
data/
├── train/
│   ├── aadhaar/     ← Add Aadhaar images here (.jpg, .png)
│   ├── pan/         ← Add PAN images here
│   ├── fake/        ← Add fake document images here
│   └── other/       ← Add other document types here
├── val/
│   ├── aadhaar/     ← Validation Aadhaar images
│   ├── pan/         ← Validation PAN images
│   ├── fake/        ← Validation fake images
│   └── other/       ← Validation other images
└── test/
    ├── aadhaar/     ← Test Aadhaar images
    ├── pan/         ← Test PAN images
    ├── fake/        ← Test fake images
    └── other/       ← Test other images
```

### Recommended Split:
- **Train:** 70% of your images
- **Val:** 15% of your images
- **Test:** 15% of your images

---

## 🛠️ Helper Scripts

### Check Current Dataset Status:
```powershell
.\venv\Scripts\Activate.ps1
python prepare_dataset.py --count
```

### Verify Dataset:
```powershell
python prepare_dataset.py --count --verify
```

### Split Dataset Automatically:
```powershell
# If you have all images in one folder, split them
python create_validation_set.py
```

---

## ✅ Quick Start: Add Images Manually

### Step 1: Collect Images
- Gather Aadhaar, PAN, fake, and other document images
- Save them to a temporary folder

### Step 2: Copy to Data Folders
```powershell
# Example: Copy Aadhaar images
Copy-Item "C:\path\to\aadhaar_images\*.jpg" "data\train\aadhaar\"
Copy-Item "C:\path\to\aadhaar_images\*.png" "data\train\aadhaar\"

# Copy PAN images
Copy-Item "C:\path\to\pan_images\*.jpg" "data\train\pan\"

# Copy fake images
Copy-Item "C:\path\to\fake_images\*.jpg" "data\train\fake\"
```

### Step 3: Split into Train/Val/Test
Manually move some images:
- Keep ~70% in `train/`
- Move ~15% to `val/`
- Move ~15% to `test/`

Or use the helper script:
```powershell
python create_validation_set.py
```

### Step 4: Verify
```powershell
python prepare_dataset.py --count --verify
```

---

## 📊 Image Requirements

### Format:
- **Supported:** JPG, JPEG, PNG
- **Recommended:** JPG (smaller file size)

### Size:
- **Minimum:** 150x150 pixels
- **Recommended:** 300x300 pixels or larger
- **Note:** Images will be resized to 150x150 during training

### Quality:
- Clear, readable text
- Good lighting
- Minimal blur
- Proper orientation (not rotated)

---

## 🚨 Important Notes

1. **Privacy:** If using real documents, ensure compliance with data privacy laws
2. **Sensitive Data:** Consider blurring or masking sensitive information (Aadhaar numbers, PAN numbers)
3. **Legal:** Only use documents you have permission to use
4. **Balance:** Try to have roughly equal numbers of images per class

---

## 🎯 Next Steps After Adding Images

1. **Verify Dataset:**
   ```powershell
   python prepare_dataset.py --count --verify
   ```

2. **Train Position Detector (Optional):**
   ```powershell
   python train_positions.py --method images --input_dir data/train/aadhaar/ --doc_type aadhaar
   python train_positions.py --method images --input_dir data/train/pan/ --doc_type pan
   ```

3. **Start Training:**
   ```powershell
   python src/train.py --data_dir data --epochs 10 --batch_size 32
   ```

---

## 💡 Tips

- **Start Small:** Begin with minimum dataset to test the pipeline
- **Iterate:** Add more images as you train and evaluate
- **Diversity:** Include various lighting conditions, angles, and document states
- **Quality over Quantity:** Better to have 50 good images than 200 poor ones

---

**Status:** Ready to collect images! 📸

