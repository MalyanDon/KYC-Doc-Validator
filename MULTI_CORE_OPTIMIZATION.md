# Multi-Core Training Optimization

## 🔍 Current Status

### Your System:
- **Physical Cores:** Check with `sysctl -n hw.physicalcpu`
- **Logical Cores:** Check with `sysctl -n hw.logicalcpu`
- **TensorFlow Configuration:** Automatically uses available cores

---

## ✅ What We've Configured

### Multi-Core Settings:
```python
# Automatically detected and set:
TF_NUM_INTEROP_THREADS = num_cores  # Parallel operations between ops
TF_NUM_INTRAOP_THREADS = num_cores  # Parallelism within operations
OMP_NUM_THREADS = num_cores         # OpenMP threads
```

### What This Means:
- ✅ **Inter-op threads:** Parallel execution of independent operations
- ✅ **Intra-op threads:** Parallel execution within a single operation (matrix multiplication, etc.)
- ✅ **OpenMP threads:** Used by underlying libraries (BLAS, etc.)

---

## 📊 How to Check Core Usage

### During Training:
```bash
# Check CPU usage
top -l 1 | grep "CPU usage"

# Or use Activity Monitor (GUI)
# Look for Python process using multiple cores
```

### Check TensorFlow Configuration:
```python
import tensorflow as tf
print("Inter-op threads:", tf.config.threading.get_inter_op_parallelism_threads())
print("Intra-op threads:", tf.config.threading.get_intra_op_parallelism_threads())
```

---

## 🚀 Performance Tips

### 1. **Batch Size**
- Larger batch size = better GPU/CPU utilization
- Current: 16 (good for testing)
- Recommended: 32-64 (if memory allows)

### 2. **Data Loading**
- Current: Custom generator (single-threaded)
- Could optimize with `tf.data` for parallel loading

### 3. **Model Operations**
- TensorFlow automatically parallelizes:
  - Matrix multiplications
  - Convolutions
  - Batch operations

---

## ⚡ Optimization Options

### Option 1: Increase Batch Size (Recommended)
```bash
# Current
python src/train.py --batch_size 16

# Optimized (if you have enough RAM)
python src/train.py --batch_size 32
# or
python src/train.py --batch_size 64
```

### Option 2: Use tf.data (Advanced)
Replace custom generator with `tf.data.Dataset` for:
- Parallel data loading
- Prefetching
- Better memory management

### Option 3: Mixed Precision (If GPU Available)
```python
# Use float16 for faster computation
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

---

## 📈 Expected Performance

### Single Core:
- ~1-2 images/second
- Training time: Very slow

### Multi-Core (Your Setup):
- ~4-8 images/second (depending on CPU)
- Training time: Much faster

### With GPU (If Available):
- ~50-200 images/second
- Training time: Very fast

---

## 🔍 Monitor Core Usage

### Real-time Monitoring:
```bash
# Terminal 1: Run training
python src/train.py --data_dir data --epochs 10

# Terminal 2: Monitor CPU
watch -n 1 'top -l 1 | grep "CPU usage"'
```

### Check Process:
```bash
# See which cores Python is using
ps -M -p $(pgrep -f train.py)
```

---

## 💡 Current Configuration

**Your training script now:**
- ✅ Automatically detects CPU cores
- ✅ Sets TensorFlow to use all cores
- ✅ Optimizes thread configuration
- ✅ Uses parallel operations

**To verify it's working:**
1. Start training
2. Open Activity Monitor
3. Look for Python process
4. Check "CPU" column - should show high usage across multiple cores

---

## 🎯 Summary

**Yes, we are using multiple cores!**

The training script automatically:
- Detects your CPU cores
- Configures TensorFlow to use all cores
- Parallelizes operations

**To maximize performance:**
- Use larger batch size (32-64)
- Ensure enough RAM
- Monitor CPU usage during training

---

**Your training is optimized for multi-core usage!** 🚀

