# Validation Set Guide - What to Add

## 🎯 What is Validation Set?

**Validation set** is used during training to:
- ✅ Check if model is learning (not just memorizing)
- ✅ Stop training early if not improving
- ✅ Tune hyperparameters
- ✅ Monitor for overfitting

**It's NOT used for:**
- ❌ Training the model
- ❌ Final testing (that's test set)

---

## 📊 What Goes in Validation Set?

### Simple Answer:
**Move 15% of your training images to validation set**

### For Your Dataset:

**Current:**
- Train Aadhaar: 1,852 images
- Test Aadhaar: 265 images
- Val Aadhaar: 0 images

**After Split:**
- Train Aadhaar: ~1,574 images (85%)
- Val Aadhaar: ~278 images (15%)
- Test Aadhaar: 265 images (unchanged)

---

## 🔄 How to Create Validation Set

### Method 1: Automatic (Recommended)

```bash
# Run the script I just created
python create_validation_set.py --data_dir data --split_ratio 0.15
```

**What it does:**
- Takes 15% of images from `data/train/aadhaar/`
- Moves them to `data/val/aadhaar/`
- Does this for all classes (aadhaar, pan, fake, other)
- Randomly selects images (so it's representative)

### Method 2: Manual

```bash
# Manually move some images
mkdir -p data/val/aadhaar
# Move ~200-300 images from train/aadhaar to val/aadhaar
```

---

## 📐 Recommended Split Ratios

### Standard Split:
- **Train:** 70-80% (for learning)
- **Val:** 10-15% (for validation during training)
- **Test:** 10-15% (for final evaluation)

### For Your Dataset (2,117 images):
- **Train:** ~1,574 images (74%)
- **Val:** ~278 images (13%)
- **Test:** 265 images (13%)

**This is a good split!**

---

## ✅ What Images to Move?

### Best Practice:
- ✅ **Random selection** - Don't pick specific images
- ✅ **Representative** - Should match train distribution
- ✅ **Same quality** - Mix of good/bad quality images
- ✅ **Same format** - Same document types

### Don't:
- ❌ Move only "good" images
- ❌ Move only "bad" images
- ❌ Move in order (first/last images)
- ❌ Move all from one source

**The script does random selection automatically!**

---

## 🎯 After Creating Validation Set

### Verify:
```bash
python prepare_dataset.py --count --verify
```

**Expected output:**
```
TRAIN:
  aadhaar: ~1,574 images

VAL:
  aadhaar: ~278 images

TEST:
  aadhaar: 265 images
```

### Then Train:
```bash
python src/train.py --data_dir data --epochs 10
```

---

## 📋 Summary

**What to add in validation set:**
- ✅ 15% of your training images
- ✅ Randomly selected
- ✅ Same classes as training (aadhaar, pan, fake, other)
- ✅ Representative of training data

**How to create:**
```bash
python create_validation_set.py
```

**That's it!** The script does everything automatically. 🚀

