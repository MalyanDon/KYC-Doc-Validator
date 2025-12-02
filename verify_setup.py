"""
Setup Verification Script
Verifies that all dependencies are installed and configured correctly
"""

import sys
import os

def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def check_python_packages():
    """Check if all required Python packages are installed"""
    print_header("Checking Python Packages")
    
    required_packages = {
        'tensorflow': 'TensorFlow',
        'keras': 'Keras',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'pytesseract': 'pytesseract',
        'pyzbar': 'pyzbar',
        'albumentations': 'albumentations',
        'fitz': 'PyMuPDF',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'pandas': 'pandas',
        'streamlit': 'Streamlit',
        'requests': 'requests',
        'tqdm': 'tqdm'
    }
    
    all_ok = True
    for module_name, package_name in required_packages.items():
        try:
            if module_name == 'cv2':
                import cv2
                print(f"[OK] {package_name}: {cv2.__version__}")
            elif module_name == 'PIL':
                from PIL import Image
                import PIL
                print(f"[OK] {package_name}: {PIL.__version__}")
            elif module_name == 'fitz':
                import fitz
                print(f"[OK] {package_name}: {fitz.version[0]}")
            elif module_name == 'sklearn':
                import sklearn
                print(f"[OK] {package_name}: {sklearn.__version__}")
            else:
                module = __import__(module_name)
                version = getattr(module, '__version__', 'installed')
                print(f"[OK] {package_name}: {version}")
        except ImportError:
            print(f"[FAIL] {package_name}: Not installed")
            all_ok = False
    
    return all_ok

def check_tesseract():
    """Check if Tesseract OCR is installed and accessible"""
    print_header("Checking Tesseract OCR")
    
    try:
        import pytesseract
        import shutil
        
        # Try to find Tesseract
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        
        found = False
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                found = True
                break
        
        if not found:
            tesseract_cmd = shutil.which('tesseract')
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                found = True
        
        if found:
            # Try to get version
            try:
                version = pytesseract.get_tesseract_version()
                print(f"[OK] Tesseract OCR: v{version}")
                print(f"     Path: {pytesseract.pytesseract.tesseract_cmd}")
                return True
            except Exception as e:
                print(f"[WARN] Tesseract found but version check failed: {e}")
                return True
        else:
            print("[FAIL] Tesseract OCR not found")
            print("       Install from: https://github.com/UB-Mannheim/tesseract/wiki")
            return False
            
    except ImportError:
        print("[FAIL] pytesseract not installed")
        return False

def check_model_files():
    """Check if model files exist"""
    print_header("Checking Model Files")
    
    model_path = 'models/kyc_validator.h5'
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"[OK] Trained model found: {model_path} ({size:.2f} MB)")
        return True
    else:
        print(f"[INFO] Trained model not found: {model_path}")
        print("       Model needs to be trained with: python src/train.py")
        return False

def check_data_structure():
    """Check if data directory structure exists"""
    print_header("Checking Data Directory Structure")
    
    required_dirs = [
        'data/train/aadhaar',
        'data/train/pan',
        'data/train/fake',
        'data/train/other',
        'data/val/aadhaar',
        'data/val/pan',
        'data/val/fake',
        'data/val/other',
        'data/test/aadhaar',
        'data/test/pan',
        'data/test/fake',
        'data/test/other',
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            # Count images
            image_count = len([f for f in os.listdir(dir_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            status = f"{image_count} images" if image_count > 0 else "empty"
            print(f"[OK] {dir_path}: {status}")
        else:
            print(f"[FAIL] {dir_path}: Missing")
            all_exist = False
    
    return all_exist

def test_model_creation():
    """Test if model can be created (will download VGG16 weights)"""
    print_header("Testing Model Creation")
    
    try:
        sys.path.insert(0, 'src')
        from models import create_ensemble_model, compile_model
        
        print("Creating ensemble model...")
        model = create_ensemble_model(input_shape=(150, 150, 3), num_classes=4)
        compile_model(model, learning_rate=0.001)
        
        param_count = model.count_params()
        print(f"[OK] Model created successfully!")
        print(f"     Total parameters: {param_count:,}")
        print(f"     Model architecture: Ensemble (VGG16 + Custom CNN + Sequential)")
        return True
    except Exception as e:
        print(f"[FAIL] Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 60)
    print("KYC Document Validator - Setup Verification")
    print("=" * 60)
    
    results = {
        'Python Packages': check_python_packages(),
        'Tesseract OCR': check_tesseract(),
        'Data Structure': check_data_structure(),
        'Model Creation': test_model_creation(),
        'Trained Model': check_model_files(),
    }
    
    print_header("Summary")
    all_ok = True
    for component, status in results.items():
        status_text = "[OK]" if status else "[INFO/FAIL]"
        print(f"{status_text} {component}")
        if not status and component in ['Trained Model', 'Data Structure']:
            # These are expected to be missing initially
            pass
        elif not status:
            all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("[SUCCESS] All critical components are ready!")
        print("\nNext steps:")
        print("1. Add images to data/ directories")
        print("2. Train model: python src/train.py --data_dir data --epochs 10")
        print("3. Run app: streamlit run app/streamlit_app.py")
    else:
        print("[WARNING] Some components need attention (see above)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()

