# Training Steps - What to Do Now

## 🎯 Current Status

✅ Virtual environment: Created  
✅ Dataset structure: Created  
✅ Training scripts: Ready  
⚠️ **Dataset: Empty (0 images)** ← **You need to add images!**

---

## 📋 Step-by-Step: What to Do

### **STEP 1: Add Images to Dataset** (Most Important!)

You need to add images to these folders:

```bash
# Navigate to project
cd "/Users/abhishekmalyan/Aadhar Card and Pancard/KYC-Doc-Validator"

# Check current status
python prepare_dataset.py --count
```

**Add images to:**
- `data/train/aadhaar/` - Real Aadhaar card images
- `data/train/pan/` - Real PAN card images  
- `data/train/fake/` - Fake/synthetic documents
- `data/train/other/` - Other document types

**Also add to:**
- `data/val/` - Same structure (validation set)
- `data/test/` - Same structure (test set)

**How to add:**
```bash
# Option 1: Copy images manually
cp /path/to/your/aadhaar_images/*.jpg data/train/aadhaar/
cp /path/to/your/pan_images/*.jpg data/train/pan/

# Option 2: Use file manager to drag and drop images
# Option 3: Download from sources and organize
```

**Minimum:** 10-20 images per class per split  
**Recommended:** 50+ images per class

---

### **STEP 2: Verify Dataset**

```bash
# Check if images are added
python prepare_dataset.py --count --verify
```

**You should see:**
```
TRAIN:
  aadhaar:   20 images  ← Should be > 0
  pan:       20 images  ← Should be > 0
  fake:      15 images  ← Should be > 0
  ...
```

---

### **STEP 3: Train Position Detector (Optional but Recommended)**

**Quick auto-learn:**
```bash
# Activate environment
source venv/bin/activate

# Learn Aadhaar positions
python train_positions.py \
    --method images \
    --input_dir data/train/aadhaar/ \
    --doc_type aadhaar

# Learn PAN positions  
python train_positions.py \
    --method images \
    --input_dir data/train/pan/ \
    --doc_type pan
```

**Time:** ~5-10 minutes

---

### **STEP 4: Train Main Model**

```bash
# Make sure environment is activated
source venv/bin/activate

# Start training
python src/train.py --data_dir data --epochs 10
```

**What to expect:**
- Training will start
- You'll see progress bars
- Model saved to `models/kyc_validator.h5`
- Confusion matrix and plots generated

**Time:** 30-60 minutes (depending on dataset size)

---

### **STEP 5: Test Your Model**

```bash
# Test with command line
python test_model.py --image data/test/aadhaar/sample.jpg

# Or use Streamlit
streamlit run app/streamlit_app.py
```

---

## 🚀 Quick Start (Copy-Paste Commands)

```bash
# 1. Navigate to project
cd "/Users/abhishekmalyan/Aadhar Card and Pancard/KYC-Doc-Validator"

# 2. Activate environment
source venv/bin/activate

# 3. Verify dataset structure
python prepare_dataset.py --count

# 4. ADD YOUR IMAGES HERE (manually copy to folders)
#    - data/train/aadhaar/
#    - data/train/pan/
#    - data/train/fake/
#    - data/val/...
#    - data/test/...

# 5. Verify images are added
python prepare_dataset.py --count --verify

# 6. Train position detector (optional)
python train_positions.py --method images --input_dir data/train/aadhaar/ --doc_type aadhaar

# 7. Train main model
python src/train.py --data_dir data --epochs 10

# 8. Test
python test_model.py --image data/test/aadhaar/sample.jpg
```

---

## 📦 Where to Get Images?

### Option 1: Use Your Own (Best)
- Real Aadhaar/PAN cards (with permission)
- Scan or photograph documents
- Organize into folders

### Option 2: Check Cloned Repositories
```bash
# Check if cloned repos have data
find temp_magnum -name "*.jpg" -o -name "*.png" | head -10
find temp_doc -name "*.jpg" -o -name "*.png" | head -10

# If found, copy to your dataset
# cp temp_magnum/data/*.jpg data/train/aadhaar/
```

### Option 3: Create Synthetic Fakes
- Use real documents as base
- Apply augmentations to create variations
- Label as "fake" class

### Option 4: Start Small (For Testing)
- Use 5-10 images per class initially
- Test the pipeline
- Add more data later

---

## ⚠️ Important Notes

1. **You MUST add images first** - Training won't work without data
2. **Minimum 10-20 images per class** - For basic training
3. **Split into train/val/test** - 70/15/15 ratio recommended
4. **Image formats:** JPG, JPEG, PNG
5. **Image size:** Any size (will be resized to 150x150)

---

## 🎯 What Happens During Training

```
1. Loads images from data/train/
   ↓
2. Preprocesses (resize to 150x150, normalize)
   ↓
3. Creates ensemble model (VGG16 + Custom CNN + Sequential)
   ↓
4. Trains for 10 epochs
   - Shows progress bars
   - Validates on data/val/
   ↓
5. Saves best model to models/kyc_validator.h5
   ↓
6. Evaluates on data/test/
   ↓
7. Generates confusion_matrix.png and training_history.png
```

---

## ✅ Success Indicators

After training, you should have:
- ✅ `models/kyc_validator.h5` - Trained model (100-500 MB)
- ✅ `confusion_matrix.png` - Performance visualization
- ✅ `training_history.png` - Training curves
- ✅ Terminal shows final accuracy metrics

---

## 🆘 If You Don't Have Images Yet

**For Testing/Development:**
1. Create a few sample images manually
2. Or use the cloned repositories if they have data
3. Start with 5-10 images per class
4. Test the pipeline
5. Add more real data later

**Quick Test Setup:**
```bash
# Create a few test images (you can use any images for testing)
# Just to verify the pipeline works

# Then train with small dataset
python src/train.py --data_dir data --epochs 5 --batch_size 8
```

---

## 📚 Documentation

- **`START_TRAINING.md`** - Detailed training guide
- **`TRAINING_GUIDE.md`** - Complete guide with troubleshooting
- **`QUICK_START.md`** - 5-minute quick start

---

**Your next step: ADD IMAGES to the dataset folders!** 📸

Then run: `python src/train.py --data_dir data --epochs 10`

